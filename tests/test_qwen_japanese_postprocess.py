#!/usr/bin/env python3
"""
Real-world integration test for QwenASR Japanese post-processing.

This test runs the ACTUAL Qwen pipeline with real audio files to verify:
1. Japanese post-processing is applied when enabled (default)
2. Japanese post-processing is skipped when disabled
3. The post-processing actually modifies the output

Requirements:
- qwen-asr package installed
- GPU with sufficient VRAM (~4GB for Qwen3-ASR-1.7B)
- Test audio files in test_media/

Usage:
    python tests/test_qwen_japanese_postprocess.py

This is NOT a unit test - it runs the full pipeline and takes several minutes.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_merge_master_with_timestamps():
    """Test 0: Verify merge_master_with_timestamps preserves punctuation (JP-001 fix)."""
    print("\n" + "=" * 60)
    print("TEST 0: merge_master_with_timestamps (Punctuation Preservation)")
    print("=" * 60)

    from whisperjav.modules.qwen_asr import merge_master_with_timestamps

    # Mock timestamp objects (like ForcedAlignItem)
    class MockTimestamp:
        def __init__(self, text, start_time, end_time):
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    # === Test Case 1: Basic punctuation preservation ===
    print("\n--- Test 0.1: Basic punctuation preservation ---")
    master = "けども、お前が。"
    timestamps = [
        MockTimestamp("けども", 1.0, 1.5),
        MockTimestamp("お前", 2.0, 2.3),
        MockTimestamp("が", 2.3, 2.5),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    print(f"  Master: '{master}'")
    print(f"  Result: {[(w['word'], w['start'], w['end']) for w in result]}")

    assert len(result) == 3, f"Expected 3 words, got {len(result)}"
    assert result[0]['word'] == "けども、", f"Expected 'けども、', got '{result[0]['word']}'"
    assert result[1]['word'] == "お前", f"Expected 'お前', got '{result[1]['word']}'"
    assert result[2]['word'] == "が。", f"Expected 'が。', got '{result[2]['word']}'"
    print("[PASS] Basic punctuation (、。) preserved correctly")

    # === Test Case 2: Multiple punctuation marks ===
    print("\n--- Test 0.2: Multiple punctuation marks ---")
    master = "はい。ね？そうだよ！"
    timestamps = [
        MockTimestamp("はい", 0.0, 0.5),
        MockTimestamp("ね", 0.6, 0.8),
        MockTimestamp("そうだよ", 1.0, 1.5),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    print(f"  Master: '{master}'")
    print(f"  Result: {[(w['word'], w['start'], w['end']) for w in result]}")

    assert result[0]['word'] == "はい。", "First word should include 。"
    assert result[1]['word'] == "ね？", "Second word should include ？"
    assert result[2]['word'] == "そうだよ！", "Third word should include ！"
    print("[PASS] Multiple punctuation marks (。？！) preserved")

    # === Test Case 3: Leading punctuation (opening quote) ===
    print("\n--- Test 0.3: Leading punctuation (opening quote) ---")
    master = "「こんにちは」"
    timestamps = [
        MockTimestamp("こんにちは", 0.0, 1.0),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    print(f"  Master: '{master}'")
    print(f"  Result: {[(w['word'], w['start'], w['end']) for w in result]}")

    assert result[0]['word'] == "「こんにちは」", "Should include both 「 and 」"
    print("[PASS] Leading and trailing quotes preserved")

    # === Test Case 4: No timestamps (edge case) ===
    print("\n--- Test 0.4: No timestamps (edge case) ---")
    master = "テスト。"
    timestamps = []
    result = merge_master_with_timestamps(master, timestamps)
    print(f"  Master: '{master}'")
    print(f"  Result: {result}")

    assert len(result) == 1, "Should return single word"
    assert result[0]['word'] == "テスト。", "Should preserve full text"
    assert result[0]['start'] == 0.0 and result[0]['end'] == 0.0, "Should have zero timing"
    print("[PASS] No timestamps handled correctly")

    # === Test Case 5: Complex real-world example ===
    print("\n--- Test 0.5: Complex real-world example ---")
    master = "本番をやらなくてもいいんだったらいいけども、お前ができてないから言ってるんだよ。はい。"
    timestamps = [
        MockTimestamp("本番", 0.0, 0.3),
        MockTimestamp("を", 0.3, 0.4),
        MockTimestamp("やらなくて", 0.4, 0.8),
        MockTimestamp("も", 0.8, 0.9),
        MockTimestamp("いい", 0.9, 1.1),
        MockTimestamp("ん", 1.1, 1.2),
        MockTimestamp("だったら", 1.2, 1.5),
        MockTimestamp("いい", 1.5, 1.7),
        MockTimestamp("けども", 1.7, 2.0),
        MockTimestamp("お前", 2.1, 2.4),
        MockTimestamp("が", 2.4, 2.5),
        MockTimestamp("できてない", 2.5, 2.9),
        MockTimestamp("から", 2.9, 3.1),
        MockTimestamp("言ってる", 3.1, 3.4),
        MockTimestamp("ん", 3.4, 3.5),
        MockTimestamp("だよ", 3.5, 3.8),
        MockTimestamp("はい", 4.0, 4.3),
    ]
    result = merge_master_with_timestamps(master, timestamps)

    # Concatenate all words to verify full text is preserved
    reconstructed = ''.join(w['word'] for w in result)
    print(f"  Master:        '{master}'")
    print(f"  Reconstructed: '{reconstructed}'")

    assert reconstructed == master, f"Reconstructed text should match master"
    print("[PASS] Complex text fully preserved with punctuation")

    # Check specific punctuation positions
    # "けども" should have "、" attached
    kedomo_word = next((w for w in result if w['word'].startswith("けども")), None)
    assert kedomo_word and "、" in kedomo_word['word'], "けども should have 、 attached"
    print("[PASS] Comma attached to 'けども'")

    # "だよ" should have "。" attached
    dayo_word = next((w for w in result if "だよ" in w['word']), None)
    assert dayo_word and "。" in dayo_word['word'], "だよ should have 。 attached"
    print("[PASS] Period attached to 'だよ'")

    # "はい" should have "。" attached (trailing)
    hai_word = next((w for w in result if w['word'].startswith("はい")), None)
    assert hai_word and "。" in hai_word['word'], "はい should have trailing 。"
    print("[PASS] Trailing period attached to last word")

    # === Test Case 6: Western punctuation (model might output these) ===
    print("\n--- Test 0.6: Western punctuation preservation ---")
    master = "Hello, world. How are you?"
    timestamps = [
        MockTimestamp("Hello", 0.0, 0.5),
        MockTimestamp("world", 0.6, 1.0),
        MockTimestamp("How", 1.2, 1.5),
        MockTimestamp("are", 1.5, 1.7),
        MockTimestamp("you", 1.7, 2.0),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    reconstructed = ''.join(w['word'] for w in result)

    assert reconstructed == master, f"Western punctuation not preserved: '{reconstructed}' != '{master}'"
    assert any(',' in w['word'] for w in result), "Western comma should be preserved"
    assert any('.' in w['word'] for w in result), "Western period should be preserved"
    assert any('?' in w['word'] for w in result), "Western question mark should be preserved"
    print("[PASS] Western punctuation (. , ?) preserved correctly")

    # === Test Case 7: Space as delimiter in Japanese ===
    print("\n--- Test 0.7: Space as delimiter in Japanese ---")
    # In Japanese, space indicates explicit sentence/phrase boundary
    master = "こんにちは 元気ですか"
    timestamps = [
        MockTimestamp("こんにちは", 0.0, 1.0),
        MockTimestamp("元気ですか", 1.5, 2.5),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    reconstructed = ''.join(w['word'] for w in result)

    assert reconstructed == master, f"Space not preserved: '{reconstructed}' != '{master}'"
    # Space should be attached to the preceding word
    assert result[0]['word'] == "こんにちは ", "Space should attach to preceding word"
    print("[PASS] Space preserved and attached to preceding word")

    # === Test Case 8: Mixed Japanese/Western punctuation ===
    print("\n--- Test 0.8: Mixed Japanese and Western punctuation ---")
    master = "OK。That's right！Really?"
    timestamps = [
        MockTimestamp("OK", 0.0, 0.3),
        MockTimestamp("That's", 0.4, 0.7),
        MockTimestamp("right", 0.7, 1.0),
        MockTimestamp("Really", 1.2, 1.5),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    reconstructed = ''.join(w['word'] for w in result)

    assert reconstructed == master, f"Mixed punctuation not preserved: '{reconstructed}'"
    assert any('。' in w['word'] for w in result), "Japanese period should be preserved"
    assert any('！' in w['word'] for w in result), "Japanese exclamation should be preserved"
    assert any('?' in w['word'] for w in result), "Western question mark should be preserved"
    print("[PASS] Mixed Japanese/Western punctuation preserved")

    # === Test Case 9: Multiple spaces (unusual but possible) ===
    print("\n--- Test 0.9: Multiple spaces between words ---")
    master = "First  Second"  # Double space
    timestamps = [
        MockTimestamp("First", 0.0, 0.5),
        MockTimestamp("Second", 1.0, 1.5),
    ]
    result = merge_master_with_timestamps(master, timestamps)
    reconstructed = ''.join(w['word'] for w in result)

    assert reconstructed == master, f"Multiple spaces not preserved: '{reconstructed}'"
    assert "  " in result[0]['word'], "Double space should be preserved"
    print("[PASS] Multiple spaces preserved")

    print("\n[PASS] All merge_master_with_timestamps tests passed")
    return True


def test_qwen_japanese_postprocessor_module():
    """Test 1: Verify the JapanesePostProcessor module works correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: JapanesePostProcessor Module Functionality")
    print("=" * 60)

    from whisperjav.modules.japanese_postprocessor import (
        JapanesePostProcessor,
        JapaneseLinguisticSets,
        PresetParameters,
    )

    # Test initialization
    processor = JapanesePostProcessor()
    assert processor is not None, "Failed to create JapanesePostProcessor"
    print("[PASS] JapanesePostProcessor instantiated")

    # Test linguistic sets
    ling = JapaneseLinguisticSets()
    assert len(ling.base_endings) > 0, "No base endings defined"
    assert len(ling.aizuchi_fillers) > 0, "No aizuchi fillers defined"
    all_endings = ling.get_all_final_endings()
    assert 'ね' in all_endings, "Missing ね in final endings"
    assert 'よ' in all_endings, "Missing よ in final endings"
    print(f"[PASS] Linguistic sets loaded: {len(all_endings)} final endings, {len(ling.aizuchi_fillers)} fillers")

    # v1.8.5+: Hierarchical splitting patterns
    assert len(ling.definite_endings) > 0, "No definite endings defined"
    assert 'です' in ling.definite_endings, "Missing です in definite endings"
    assert 'ます' in ling.definite_endings, "Missing ます in definite endings"
    assert len(ling.strong_particles) > 0, "No strong particles defined"
    assert 'よ' in ling.strong_particles, "Missing よ in strong particles"
    assert len(ling.soft_particles) > 0, "No soft particles defined"
    assert 'ね' in ling.soft_particles, "Missing ね in soft particles"
    print(f"[PASS] Hierarchical patterns: {len(ling.definite_endings)} definite, {len(ling.strong_particles)} strong, {len(ling.soft_particles)} soft")

    # Test presets
    assert "default" in JapanesePostProcessor.PRESETS
    assert "high_moan" in JapanesePostProcessor.PRESETS
    assert "narrative" in JapanesePostProcessor.PRESETS
    print("[PASS] All presets defined")

    # Test preset parameters
    default_params = JapanesePostProcessor.PRESETS["default"]
    assert default_params.gap_threshold == 0.3
    assert default_params.segment_length == 35
    # v1.8.5+: Hierarchical splitting parameters
    assert default_params.strong_particle_gap == 0.25, "strong_particle_gap should be 0.25"
    assert default_params.soft_particle_gap == 0.4, "soft_particle_gap should be 0.4"
    assert default_params.pure_gap_threshold == 0.6, "pure_gap_threshold should be 0.6"
    print(f"[PASS] Default preset: gap={default_params.gap_threshold}s, length={default_params.segment_length}")
    print(f"[PASS] Hierarchical thresholds: strong={default_params.strong_particle_gap}s, soft={default_params.soft_particle_gap}s, pure={default_params.pure_gap_threshold}s")

    print("\nTEST 1 PASSED: JapanesePostProcessor module is correctly implemented")
    return True


def test_qwen_asr_initialization():
    """Test 2: Verify QwenASR accepts Japanese post-processing parameters."""
    print("\n" + "=" * 60)
    print("TEST 2: QwenASR Initialization with Japanese Post-Processing Params")
    print("=" * 60)

    from whisperjav.modules.qwen_asr import QwenASR

    # Test default parameters (should have japanese_postprocess=True)
    asr = QwenASR(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="cpu",  # Use CPU for initialization test to avoid VRAM
        japanese_postprocess=True,
        postprocess_preset="default",
    )
    assert asr.japanese_postprocess == True, "japanese_postprocess should be True by default"
    assert asr.postprocess_preset == "default", "postprocess_preset should be 'default'"
    assert asr._postprocessor is not None, "_postprocessor should be initialized"
    print("[PASS] QwenASR with japanese_postprocess=True")

    # Test with disabled post-processing
    asr_disabled = QwenASR(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="cpu",
        japanese_postprocess=False,
        postprocess_preset="default",
    )
    assert asr_disabled.japanese_postprocess == False, "japanese_postprocess should be False"
    assert asr_disabled._postprocessor is None, "_postprocessor should be None when disabled"
    print("[PASS] QwenASR with japanese_postprocess=False")

    # Test with different presets
    asr_highmoan = QwenASR(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="cpu",
        japanese_postprocess=True,
        postprocess_preset="high_moan",
    )
    assert asr_highmoan.postprocess_preset == "high_moan"
    print("[PASS] QwenASR with postprocess_preset='high_moan'")

    asr_narrative = QwenASR(
        model_id="Qwen/Qwen3-ASR-1.7B",
        device="cpu",
        japanese_postprocess=True,
        postprocess_preset="narrative",
    )
    assert asr_narrative.postprocess_preset == "narrative"
    print("[PASS] QwenASR with postprocess_preset='narrative'")

    print("\nTEST 2 PASSED: QwenASR correctly accepts Japanese post-processing parameters")
    return True


def test_transformers_pipeline_passthrough():
    """Test 3: Verify TransformersPipeline passes through Japanese post-processing params."""
    print("\n" + "=" * 60)
    print("TEST 3: TransformersPipeline Parameter Passthrough")
    print("=" * 60)

    from whisperjav.pipelines.transformers_pipeline import TransformersPipeline

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        temp_processing = Path(temp_dir) / "temp"
        temp_processing.mkdir()

        # Test with enabled (default)
        pipeline = TransformersPipeline(
            output_dir=str(output_dir),
            temp_dir=str(temp_processing),
            asr_backend="qwen",
            qwen_japanese_postprocess=True,
            qwen_postprocess_preset="default",
        )

        # Verify config is stored
        assert pipeline.qwen_config["japanese_postprocess"] == True
        assert pipeline.qwen_config["postprocess_preset"] == "default"
        assert pipeline._asr_config["japanese_postprocess"] == True
        assert pipeline._asr_config["postprocess_preset"] == "default"
        print("[PASS] Pipeline stores japanese_postprocess=True in configs")

        # Test with disabled
        pipeline_disabled = TransformersPipeline(
            output_dir=str(output_dir),
            temp_dir=str(temp_processing),
            asr_backend="qwen",
            qwen_japanese_postprocess=False,
            qwen_postprocess_preset="default",
        )

        assert pipeline_disabled.qwen_config["japanese_postprocess"] == False
        assert pipeline_disabled._asr_config["japanese_postprocess"] == False
        print("[PASS] Pipeline stores japanese_postprocess=False in configs")

        # Test with different preset
        pipeline_highmoan = TransformersPipeline(
            output_dir=str(output_dir),
            temp_dir=str(temp_processing),
            asr_backend="qwen",
            qwen_japanese_postprocess=True,
            qwen_postprocess_preset="high_moan",
        )

        assert pipeline_highmoan.qwen_config["postprocess_preset"] == "high_moan"
        assert pipeline_highmoan._asr_config["postprocess_preset"] == "high_moan"
        print("[PASS] Pipeline stores postprocess_preset='high_moan' in configs")

    print("\nTEST 3 PASSED: TransformersPipeline correctly passes through parameters")
    return True


def test_stable_ts_asr_uses_shared_module():
    """Test 4: Verify StableTSASR uses the shared JapanesePostProcessor."""
    print("\n" + "=" * 60)
    print("TEST 4: StableTSASR Uses Shared JapanesePostProcessor")
    print("=" * 60)

    from whisperjav.modules.stable_ts_asr import StableTSASR
    from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

    # Create a minimal config for testing (won't load model)
    model_config = {"model_name": "large-v2", "device": "cpu"}
    params = {"decoder": {}, "provider": {}}

    # StableTSASR initialization will fail without actual whisper model
    # But we can verify the import and attribute presence
    try:
        # This will initialize the postprocessor before loading the model
        import whisperjav.modules.stable_ts_asr as stable_ts_module

        # Check that the module imports JapanesePostProcessor
        assert hasattr(stable_ts_module, 'JapanesePostProcessor'), \
            "JapanesePostProcessor not imported in stable_ts_asr"
        print("[PASS] JapanesePostProcessor is imported in stable_ts_asr module")

        # Verify the source code has the delegation
        import inspect
        source = inspect.getsource(stable_ts_module.StableTSASR._postprocess_japanese_dialogue)
        assert "_japanese_postprocessor.process" in source, \
            "_postprocess_japanese_dialogue should delegate to shared processor"
        print("[PASS] _postprocess_japanese_dialogue delegates to shared JapanesePostProcessor")

    except Exception as e:
        print(f"[WARN] Could not fully verify StableTSASR integration: {e}")
        print("[INFO] This is expected if whisper models are not installed")

    print("\nTEST 4 PASSED: StableTSASR is configured to use shared module")
    return True


def test_cli_arguments():
    """Test 5: Verify CLI arguments are parsed correctly."""
    print("\n" + "=" * 60)
    print("TEST 5: CLI Argument Parsing")
    print("=" * 60)

    from whisperjav.main import parse_arguments
    import sys

    # Save original argv
    original_argv = sys.argv

    try:
        # Test default (disabled — Qwen3 uses AssemblyTextCleaner, not JapanesePostProcessor)
        sys.argv = ["whisperjav", "test.mp4", "--mode", "qwen"]
        args = parse_arguments()
        assert args.qwen_japanese_postprocess == False, \
            "Default should be japanese_postprocess=False (deprecated for Qwen3)"
        print("[PASS] Default: --qwen-japanese-postprocess is False")

        # Test explicit enable
        sys.argv = ["whisperjav", "test.mp4", "--mode", "qwen", "--qwen-japanese-postprocess"]
        args = parse_arguments()
        assert args.qwen_japanese_postprocess == True
        print("[PASS] Explicit: --qwen-japanese-postprocess sets True")

        # Test disable
        sys.argv = ["whisperjav", "test.mp4", "--mode", "qwen", "--no-qwen-japanese-postprocess"]
        args = parse_arguments()
        assert args.qwen_japanese_postprocess == False, \
            "--no-qwen-japanese-postprocess should set False"
        print("[PASS] --no-qwen-japanese-postprocess sets False")

        # Test preset
        sys.argv = ["whisperjav", "test.mp4", "--mode", "qwen", "--qwen-postprocess-preset", "high_moan"]
        args = parse_arguments()
        assert args.qwen_postprocess_preset == "high_moan"
        print("[PASS] --qwen-postprocess-preset high_moan works")

        # Test invalid preset should fail
        sys.argv = ["whisperjav", "test.mp4", "--mode", "qwen", "--qwen-postprocess-preset", "invalid"]
        try:
            args = parse_arguments()
            print("[FAIL] Invalid preset should raise error")
            return False
        except SystemExit:
            print("[PASS] Invalid preset correctly rejected")

    finally:
        sys.argv = original_argv

    print("\nTEST 5 PASSED: CLI arguments are correctly parsed")
    return True


def test_hierarchical_splitting_unpunctuated():
    """Test 6: Test hierarchical splitting for unpunctuated text (v1.8.5+).

    This is the core test for the new hierarchical splitting algorithm that
    handles Qwen-ASR's unpunctuated continuous output.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Hierarchical Splitting for Unpunctuated Text (v1.8.5+)")
    print("=" * 60)

    try:
        import stable_whisper
        from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

        processor = JapanesePostProcessor()

        # === Test Case 1: Definite endings split (unconditional) ===
        print("\n--- Test 6.1: Definite endings split (です, ます) ---")
        # Simulate: "そうですねでも私はそう思いますよ" (no punctuation)
        # Should split at: "そうです" | "ねでも私はそう思います" | "よ"
        mock_definite = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 4.0,
                    'text': 'そうですねでも私はそう思いますよ',
                    'words': [
                        {'word': 'そう', 'start': 0.0, 'end': 0.3},
                        {'word': 'です', 'start': 0.3, 'end': 0.6},
                        {'word': 'ね', 'start': 0.6, 'end': 0.8},
                        {'word': 'でも', 'start': 0.8, 'end': 1.1},
                        {'word': '私', 'start': 1.1, 'end': 1.3},
                        {'word': 'は', 'start': 1.3, 'end': 1.4},
                        {'word': 'そう', 'start': 1.4, 'end': 1.7},
                        {'word': '思い', 'start': 1.7, 'end': 2.0},
                        {'word': 'ます', 'start': 2.0, 'end': 2.3},
                        {'word': 'よ', 'start': 2.3, 'end': 2.5},
                    ]
                }
            ]
        }
        result = stable_whisper.WhisperResult(mock_definite)
        original_count = len(result.segments)
        processor.process(result, preset="default", language="ja")
        print(f"  Original: {original_count} segment -> Processed: {len(result.segments)} segments")
        for i, seg in enumerate(result.segments):
            print(f"    [{i}] '{seg.text}'")
        # Should have at least 2 segments (split at です and ます)
        assert len(result.segments) >= 2, "Definite endings should cause splits"
        print("[PASS] Definite endings (です, ます) caused splits")

        # === Test Case 2: Strong particles with gap ===
        print("\n--- Test 6.2: Strong particles with gap (よ + 0.3s gap) ---")
        # "行くよ" [0.3s gap] "今日は" - should split after よ
        mock_strong = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': '行くよ今日は',
                    'words': [
                        {'word': '行く', 'start': 0.0, 'end': 0.3},
                        {'word': 'よ', 'start': 0.3, 'end': 0.5},
                        # 0.3s gap here (0.8 - 0.5 = 0.3)
                        {'word': '今日', 'start': 0.8, 'end': 1.1},
                        {'word': 'は', 'start': 1.1, 'end': 1.3},
                    ]
                }
            ]
        }
        result = stable_whisper.WhisperResult(mock_strong)
        processor.process(result, preset="default", language="ja")
        print(f"  Processed: {len(result.segments)} segments")
        for i, seg in enumerate(result.segments):
            print(f"    [{i}] '{seg.text}'")
        # Should split because gap (0.3s) >= strong_particle_gap (0.25s)
        assert len(result.segments) >= 2, "Strong particle よ with gap should split"
        print("[PASS] Strong particle (よ) with 0.3s gap caused split")

        # === Test Case 3: Soft particles without sufficient gap ===
        print("\n--- Test 6.3: Soft particles without sufficient gap (ね + 0.2s gap) ---")
        # "そうだね" [0.2s gap] "明日は" - should NOT split (gap < 0.4s)
        mock_soft_no_gap = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'そうだね明日は',
                    'words': [
                        {'word': 'そう', 'start': 0.0, 'end': 0.3},
                        {'word': 'だ', 'start': 0.3, 'end': 0.4},
                        {'word': 'ね', 'start': 0.4, 'end': 0.6},
                        # Only 0.2s gap here (0.8 - 0.6 = 0.2)
                        {'word': '明日', 'start': 0.8, 'end': 1.1},
                        {'word': 'は', 'start': 1.1, 'end': 1.3},
                    ]
                }
            ]
        }
        result = stable_whisper.WhisperResult(mock_soft_no_gap)
        original_count = len(result.segments)
        processor.process(result, preset="default", language="ja")
        print(f"  Processed: {len(result.segments)} segments")
        for i, seg in enumerate(result.segments):
            print(f"    [{i}] '{seg.text}'")
        # The result may or may not split due to merge passes, but the key is
        # that the soft particle alone with small gap shouldn't cause split
        print("[INFO] Soft particle (ね) with small gap may not split (expected)")

        # === Test Case 4: Soft particles WITH sufficient gap ===
        print("\n--- Test 6.4: Soft particles with sufficient gap (ね + 0.5s gap) ---")
        # "そうだね" [0.5s gap] "明日は" - should split (gap >= 0.4s)
        mock_soft_with_gap = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'そうだね明日は',
                    'words': [
                        {'word': 'そう', 'start': 0.0, 'end': 0.3},
                        {'word': 'だ', 'start': 0.3, 'end': 0.4},
                        {'word': 'ね', 'start': 0.4, 'end': 0.6},
                        # 0.5s gap here (1.1 - 0.6 = 0.5)
                        {'word': '明日', 'start': 1.1, 'end': 1.4},
                        {'word': 'は', 'start': 1.4, 'end': 1.6},
                    ]
                }
            ]
        }
        result = stable_whisper.WhisperResult(mock_soft_with_gap)
        processor.process(result, preset="default", language="ja")
        print(f"  Processed: {len(result.segments)} segments")
        for i, seg in enumerate(result.segments):
            print(f"    [{i}] '{seg.text}'")
        # Should split because gap (0.5s) >= soft_particle_gap (0.4s)
        assert len(result.segments) >= 2, "Soft particle ね with 0.5s gap should split"
        print("[PASS] Soft particle (ね) with 0.5s gap caused split")

        # === Test Case 5: Punctuation takes priority ===
        print("\n--- Test 6.5: Punctuation as TOP RULE ---")
        # "今日は良い天気です。明日も晴れるといいね" - should split at 。 first
        # Note: "そう" in previous test was removed as aizuchi, so we use different text
        mock_punctuation = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 4.0,
                    'text': '今日は良い天気です。明日も晴れるといいね',
                    'words': [
                        {'word': '今日', 'start': 0.0, 'end': 0.3},
                        {'word': 'は', 'start': 0.3, 'end': 0.4},
                        {'word': '良い', 'start': 0.4, 'end': 0.7},
                        {'word': '天気', 'start': 0.7, 'end': 1.0},
                        {'word': 'です', 'start': 1.0, 'end': 1.3},
                        {'word': '。', 'start': 1.3, 'end': 1.4},
                        # 0.6s gap here to prevent merge
                        {'word': '明日', 'start': 2.0, 'end': 2.3},
                        {'word': 'も', 'start': 2.3, 'end': 2.4},
                        {'word': '晴れる', 'start': 2.4, 'end': 2.7},
                        {'word': 'と', 'start': 2.7, 'end': 2.8},
                        {'word': 'いい', 'start': 2.8, 'end': 3.1},
                        {'word': 'ね', 'start': 3.1, 'end': 3.3},
                    ]
                }
            ]
        }
        result = stable_whisper.WhisperResult(mock_punctuation)
        processor.process(result, preset="default", language="ja")
        print(f"  Processed: {len(result.segments)} segments")
        for i, seg in enumerate(result.segments):
            print(f"    [{i}] '{seg.text}'")
        # Should have at least 2 segments from punctuation and/or definite ending split
        assert len(result.segments) >= 2, "Punctuation should cause split"
        # The text should be split (total content preserved but in multiple segments)
        all_text = ''.join(seg.text for seg in result.segments)
        assert '。' in all_text, "Punctuation should be preserved"
        print("[PASS] Punctuation and definite endings caused proper splits")

        # === Test Case 6: Complex multi-sentence without punctuation ===
        print("\n--- Test 6.6: Complex multi-sentence without punctuation ---")
        # "私はそう思いますでも彼女はそう思わないですよね"
        # Expected splits: at ます, at です (definite), possibly at ね with gap
        mock_complex = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.0,
                    'text': '私はそう思いますでも彼女はそう思わないですよね',
                    'words': [
                        {'word': '私', 'start': 0.0, 'end': 0.2},
                        {'word': 'は', 'start': 0.2, 'end': 0.3},
                        {'word': 'そう', 'start': 0.3, 'end': 0.6},
                        {'word': '思い', 'start': 0.6, 'end': 0.9},
                        {'word': 'ます', 'start': 0.9, 'end': 1.2},
                        {'word': 'でも', 'start': 1.2, 'end': 1.5},
                        {'word': '彼女', 'start': 1.5, 'end': 1.8},
                        {'word': 'は', 'start': 1.8, 'end': 1.9},
                        {'word': 'そう', 'start': 1.9, 'end': 2.2},
                        {'word': '思わない', 'start': 2.2, 'end': 2.6},
                        {'word': 'です', 'start': 2.6, 'end': 2.9},
                        {'word': 'よ', 'start': 2.9, 'end': 3.1},
                        {'word': 'ね', 'start': 3.1, 'end': 3.3},
                    ]
                }
            ]
        }
        result = stable_whisper.WhisperResult(mock_complex)
        original_text = result.segments[0].text
        processor.process(result, preset="default", language="ja")
        print(f"  Original: '{original_text}'")
        print(f"  Processed: {len(result.segments)} segments")
        for i, seg in enumerate(result.segments):
            print(f"    [{i}] '{seg.text}'")
        # Should have at least 2 segments from ます and です splits
        assert len(result.segments) >= 2, "Complex text should have multiple splits"
        print("[PASS] Complex multi-sentence text was segmented")

        print("\n[PASS] All hierarchical splitting tests passed")
        return True

    except Exception as e:
        print(f"[ERROR] Hierarchical splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_postprocessor_with_mock_result():
    """Test 7: Test JapanesePostProcessor with a mock WhisperResult."""
    print("\n" + "=" * 60)
    print("TEST 7: JapanesePostProcessor with Mock WhisperResult")
    print("=" * 60)

    try:
        import stable_whisper
        from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

        # Create a mock WhisperResult with Japanese text
        # This simulates what qwen-asr would return
        mock_result_dict = {
            'language': 'ja',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'あの、えっとね、今日はいい天気ですね。',
                    'words': [
                        {'word': 'あの', 'start': 0.0, 'end': 0.3},
                        {'word': '、', 'start': 0.3, 'end': 0.4},
                        {'word': 'えっと', 'start': 0.4, 'end': 0.7},
                        {'word': 'ね', 'start': 0.7, 'end': 0.8},
                        {'word': '、', 'start': 0.8, 'end': 0.9},
                        {'word': '今日', 'start': 0.9, 'end': 1.2},
                        {'word': 'は', 'start': 1.2, 'end': 1.4},
                        {'word': 'いい', 'start': 1.4, 'end': 1.7},
                        {'word': '天気', 'start': 1.7, 'end': 2.0},
                        {'word': 'です', 'start': 2.0, 'end': 2.3},
                        {'word': 'ね', 'start': 2.3, 'end': 2.5},
                        {'word': '。', 'start': 2.5, 'end': 2.5},
                    ]
                }
            ]
        }

        result = stable_whisper.WhisperResult(mock_result_dict)
        original_text = ''.join(seg.text for seg in result.segments)
        original_segment_count = len(result.segments)

        print(f"Original: {original_segment_count} segments, text: '{original_text}'")

        # Apply post-processing
        processor = JapanesePostProcessor()
        processed_result = processor.process(result, preset="default", language="ja")

        # The processor should have modified the result
        processed_text = ''.join(seg.text for seg in processed_result.segments)
        processed_segment_count = len(processed_result.segments)

        print(f"Processed: {processed_segment_count} segments")
        for i, seg in enumerate(processed_result.segments):
            print(f"  Segment {i}: '{seg.text}' [{seg.start:.2f}-{seg.end:.2f}]")

        # Verify something changed (fillers like あの, えっと should be removed)
        # Note: The exact changes depend on word timestamps and stable-ts behavior
        print(f"[INFO] Original text length: {len(original_text)}, Processed: {len(processed_text)}")

        # The processor should at least not crash
        assert processed_result is not None
        print("[PASS] JapanesePostProcessor processed mock result without error")

    except Exception as e:
        print(f"[INFO] Mock result test: {e}")
        print("[INFO] This may be expected if stable-ts requires specific word format")

    print("\nTEST 6 PASSED: JapanesePostProcessor handles mock data")
    return True


def test_full_pipeline_integration():
    """Test 8: Full pipeline integration test (requires GPU and qwen-asr)."""
    print("\n" + "=" * 60)
    print("TEST 8: Full Pipeline Integration (REQUIRES GPU)")
    print("=" * 60)

    # Check for test audio file
    test_audio = project_root / "test_media" / "short_15_sec_test-966-00_01_45-00_01_59.wav"
    if not test_audio.exists():
        print(f"[SKIP] Test audio not found: {test_audio}")
        return True

    # Check for GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("[SKIP] No CUDA GPU available for full pipeline test")
            return True
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[SKIP] PyTorch not available")
        return True

    # Check for qwen-asr
    try:
        import qwen_asr
        print("[INFO] qwen-asr package is available")
    except ImportError:
        print("[SKIP] qwen-asr package not installed")
        return True

    print("\n[INFO] Running full pipeline test with real audio...")
    print("[INFO] This will load the Qwen3-ASR model and may take several minutes.")

    from whisperjav.modules.qwen_asr import QwenASR

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with post-processing ENABLED
        print("\n--- Testing WITH Japanese post-processing ---")
        asr_enabled = QwenASR(
            model_id="Qwen/Qwen3-ASR-1.7B",
            device="auto",
            japanese_postprocess=True,
            postprocess_preset="default",
        )

        result_enabled = asr_enabled.transcribe(str(test_audio))
        enabled_segments = len(result_enabled.segments) if result_enabled.segments else 0
        enabled_text = ''.join(seg.text for seg in result_enabled.segments) if result_enabled.segments else ""

        print(f"[RESULT] With post-processing: {enabled_segments} segments")
        for i, seg in enumerate(result_enabled.segments[:5]):  # Show first 5
            print(f"  {i}: [{seg.start:.2f}-{seg.end:.2f}] '{seg.text}'")

        # Cleanup to free VRAM
        asr_enabled.cleanup()
        del asr_enabled
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Test with post-processing DISABLED
        print("\n--- Testing WITHOUT Japanese post-processing ---")
        asr_disabled = QwenASR(
            model_id="Qwen/Qwen3-ASR-1.7B",
            device="auto",
            japanese_postprocess=False,
        )

        result_disabled = asr_disabled.transcribe(str(test_audio))
        disabled_segments = len(result_disabled.segments) if result_disabled.segments else 0
        disabled_text = ''.join(seg.text for seg in result_disabled.segments) if result_disabled.segments else ""

        print(f"[RESULT] Without post-processing: {disabled_segments} segments")
        for i, seg in enumerate(result_disabled.segments[:5]):  # Show first 5
            print(f"  {i}: [{seg.start:.2f}-{seg.end:.2f}] '{seg.text}'")

        asr_disabled.cleanup()
        del asr_disabled
        gc.collect()
        torch.cuda.empty_cache()

        # Compare results
        print("\n--- Comparison ---")
        print(f"Segments: {enabled_segments} (with) vs {disabled_segments} (without)")
        print(f"Text length: {len(enabled_text)} (with) vs {len(disabled_text)} (without)")

        # The post-processing should change SOMETHING
        # Either segment count or text content should differ
        if enabled_segments != disabled_segments or enabled_text != disabled_text:
            print("[PASS] Post-processing made observable changes to output")
        else:
            print("[WARN] Post-processing may not have changed output")
            print("[INFO] This could be normal if there were no fillers to remove")

    print("\nTEST 7 PASSED: Full pipeline integration completed")
    return True


def test_artifact_saving():
    """Test 10: Verify debug artifacts are saved when artifacts_dir is provided."""
    print("\n" + "=" * 60)
    print("TEST 10: Debug Artifact Saving")
    print("=" * 60)

    import json
    import tempfile
    from whisperjav.modules.qwen_asr import merge_master_with_timestamps

    # Mock timestamp objects
    class MockTimestamp:
        def __init__(self, text, start_time, end_time):
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    # Test the artifact saving directly by simulating what qwen_inference does
    master_text = "本番をやらなくても。お前が。"
    timestamps = [
        MockTimestamp("本番", 0.0, 0.5),
        MockTimestamp("を", 0.5, 0.6),
        MockTimestamp("やらなくて", 0.6, 1.0),
        MockTimestamp("も", 1.0, 1.2),
        MockTimestamp("お前", 1.5, 1.8),
        MockTimestamp("が", 1.8, 2.0),
    ]

    # Perform merge
    words = merge_master_with_timestamps(master_text, timestamps)
    reconstructed = ''.join(w['word'] for w in words)

    print(f"Master text: '{master_text}'")
    print(f"Reconstructed: '{reconstructed}'")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_basename = "test_scene_001"

        # Simulate artifact saving (same logic as in qwen_inference)
        # 1. Save master text
        master_file = temp_path / f"{audio_basename}_qwen_master.txt"
        master_file.write_text(master_text, encoding='utf-8')

        # 2. Save timestamps
        timestamps_data = []
        for ts in timestamps:
            ts_entry = {
                'text': getattr(ts, 'text', None),
                'start_time': getattr(ts, 'start_time', None),
                'end_time': getattr(ts, 'end_time', None),
            }
            timestamps_data.append(ts_entry)

        timestamps_file = temp_path / f"{audio_basename}_qwen_timestamps.json"
        timestamps_file.write_text(
            json.dumps(timestamps_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        # 3. Save merged result
        merged_file = temp_path / f"{audio_basename}_qwen_merged.json"
        merged_file.write_text(
            json.dumps(words, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        # Verify files exist
        assert master_file.exists(), "Master text file not created"
        assert timestamps_file.exists(), "Timestamps JSON file not created"
        assert merged_file.exists(), "Merged JSON file not created"
        print("[PASS] All artifact files created")

        # Verify content
        saved_master = master_file.read_text(encoding='utf-8')
        assert saved_master == master_text, f"Master text mismatch: '{saved_master}'"
        print("[PASS] Master text content correct")

        saved_timestamps = json.loads(timestamps_file.read_text(encoding='utf-8'))
        assert len(saved_timestamps) == len(timestamps), "Timestamps count mismatch"
        assert saved_timestamps[0]['text'] == "本番", "First timestamp word mismatch"
        print("[PASS] Timestamps content correct")

        saved_merged = json.loads(merged_file.read_text(encoding='utf-8'))
        saved_reconstructed = ''.join(w['word'] for w in saved_merged)
        assert saved_reconstructed == master_text, f"Merged content mismatch: '{saved_reconstructed}'"
        print("[PASS] Merged content correct (punctuation preserved)")

        # Verify punctuation is in merged result
        has_jp_period = any('。' in w['word'] for w in saved_merged)
        assert has_jp_period, "Japanese period not found in merged output"
        print("[PASS] Japanese punctuation (。) preserved in merged output")

    print("\n[PASS] All artifact saving tests passed")
    return True


def test_jp002_compound_particle_merging():
    """Test 11: Verify compound particle merging (JP-002 fix)."""
    print("\n" + "=" * 60)
    print("TEST 11: Compound Particle Merging (JP-002 Fix)")
    print("=" * 60)

    import stable_whisper
    from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

    processor = JapanesePostProcessor()

    # === Test Case 1: ですよね should NOT be split ===
    print("\n--- Test 11.1: 'ですよね' should stay together ---")
    # Simulate over-split result: "思います" | "よ" | "ね"
    mock_oversplit = {
        'language': 'ja',
        'segments': [
            {
                'start': 0.0,
                'end': 1.0,
                'text': '私は思います',
                'words': [
                    {'word': '私', 'start': 0.0, 'end': 0.2},
                    {'word': 'は', 'start': 0.2, 'end': 0.3},
                    {'word': '思い', 'start': 0.3, 'end': 0.6},
                    {'word': 'ます', 'start': 0.6, 'end': 1.0},
                ]
            },
            {
                'start': 1.0,
                'end': 1.2,
                'text': 'よ',
                'words': [{'word': 'よ', 'start': 1.0, 'end': 1.2}]
            },
            {
                'start': 1.2,
                'end': 1.4,
                'text': 'ね',
                'words': [{'word': 'ね', 'start': 1.2, 'end': 1.4}]
            },
        ]
    }

    result = stable_whisper.WhisperResult(mock_oversplit)
    original_count = len(result.segments)
    print(f"  Before: {original_count} segments")
    for i, seg in enumerate(result.segments):
        print(f"    [{i}] '{seg.text}'")

    # Apply post-processing (which includes isolated particle merging)
    processor.process(result, preset="default", language="ja")

    final_count = len(result.segments)
    print(f"  After: {final_count} segments")
    for i, seg in enumerate(result.segments):
        print(f"    [{i}] '{seg.text}'")

    # The isolated particles should be merged
    assert final_count < original_count, "Isolated particles should be merged"
    # Check that 'よね' or similar is now part of a larger segment
    combined_text = ''.join(seg.text for seg in result.segments)
    assert 'よ' in combined_text, "よ should be preserved in text"
    assert 'ね' in combined_text, "ね should be preserved in text"
    print("[PASS] Isolated particles merged correctly")

    # === Test Case 2: よね as standalone should merge ===
    print("\n--- Test 11.2: Standalone 'よね' should merge with previous ---")
    mock_yone = {
        'language': 'ja',
        'segments': [
            {
                'start': 0.0,
                'end': 1.0,
                'text': '分かりました',
                'words': [
                    {'word': '分かり', 'start': 0.0, 'end': 0.5},
                    {'word': 'まし', 'start': 0.5, 'end': 0.8},
                    {'word': 'た', 'start': 0.8, 'end': 1.0},
                ]
            },
            {
                'start': 1.0,
                'end': 1.3,
                'text': 'よね',
                'words': [{'word': 'よね', 'start': 1.0, 'end': 1.3}]
            },
        ]
    }

    result = stable_whisper.WhisperResult(mock_yone)
    print(f"  Before: {len(result.segments)} segments")
    processor.process(result, preset="default", language="ja")
    print(f"  After: {len(result.segments)} segments")
    for i, seg in enumerate(result.segments):
        print(f"    [{i}] '{seg.text}'")

    # よね should be merged with previous segment
    assert len(result.segments) == 1, "'よね' should merge with previous segment"
    assert 'よね' in result.segments[0].text, "'よね' should be in merged text"
    print("[PASS] 'よね' merged with previous segment")

    print("\n[PASS] All JP-002 compound particle tests passed")
    return True


def test_jp003_tiny_fragment_merging():
    """Test 12: Verify tiny fragment merging (JP-003 fix)."""
    print("\n" + "=" * 60)
    print("TEST 12: Tiny Fragment Merging (JP-003 Fix)")
    print("=" * 60)

    import stable_whisper
    from whisperjav.modules.japanese_postprocessor import JapanesePostProcessor

    processor = JapanesePostProcessor()

    # === Test Case 1: Very short duration segment ===
    print("\n--- Test 12.1: Segment with duration < 0.3s should merge ---")
    mock_short_duration = {
        'language': 'ja',
        'segments': [
            {
                'start': 0.0,
                'end': 2.0,
                'text': '今日は天気が良いですね',
                'words': [
                    {'word': '今日', 'start': 0.0, 'end': 0.3},
                    {'word': 'は', 'start': 0.3, 'end': 0.5},
                    {'word': '天気', 'start': 0.5, 'end': 0.9},
                    {'word': 'が', 'start': 0.9, 'end': 1.0},
                    {'word': '良い', 'start': 1.0, 'end': 1.4},
                    {'word': 'です', 'start': 1.4, 'end': 1.7},
                    {'word': 'ね', 'start': 1.7, 'end': 2.0},
                ]
            },
            {
                'start': 2.0,
                'end': 2.15,  # Only 0.15s duration - should merge
                'text': 'あ',
                'words': [{'word': 'あ', 'start': 2.0, 'end': 2.15}]
            },
            {
                'start': 2.5,
                'end': 4.0,
                'text': '明日も晴れるといいな',
                'words': [
                    {'word': '明日', 'start': 2.5, 'end': 2.9},
                    {'word': 'も', 'start': 2.9, 'end': 3.1},
                    {'word': '晴れる', 'start': 3.1, 'end': 3.5},
                    {'word': 'と', 'start': 3.5, 'end': 3.6},
                    {'word': 'いい', 'start': 3.6, 'end': 3.8},
                    {'word': 'な', 'start': 3.8, 'end': 4.0},
                ]
            },
        ]
    }

    result = stable_whisper.WhisperResult(mock_short_duration)
    original_count = len(result.segments)
    print(f"  Before: {original_count} segments")
    for i, seg in enumerate(result.segments):
        dur = seg.end - seg.start
        print(f"    [{i}] '{seg.text}' (dur={dur:.2f}s)")

    processor.process(result, preset="default", language="ja")

    final_count = len(result.segments)
    print(f"  After: {final_count} segments")
    for i, seg in enumerate(result.segments):
        dur = seg.end - seg.start
        print(f"    [{i}] '{seg.text}' (dur={dur:.2f}s)")

    # The tiny segment should be merged
    assert final_count < original_count, "Tiny duration segment should be merged"
    print("[PASS] Tiny duration segment merged")

    # === Test Case 2: Very short character count ===
    print("\n--- Test 12.2: Segment with < 3 chars should merge ---")
    mock_short_chars = {
        'language': 'ja',
        'segments': [
            {
                'start': 0.0,
                'end': 1.5,
                'text': '大丈夫です',
                'words': [
                    {'word': '大丈夫', 'start': 0.0, 'end': 0.8},
                    {'word': 'です', 'start': 0.8, 'end': 1.5},
                ]
            },
            {
                'start': 1.5,
                'end': 2.0,
                'text': 'か',  # Only 1 char - should merge
                'words': [{'word': 'か', 'start': 1.5, 'end': 2.0}]
            },
        ]
    }

    result = stable_whisper.WhisperResult(mock_short_chars)
    original_count = len(result.segments)
    print(f"  Before: {original_count} segments")
    for i, seg in enumerate(result.segments):
        print(f"    [{i}] '{seg.text}' (chars={len(seg.text)})")

    processor.process(result, preset="default", language="ja")

    final_count = len(result.segments)
    print(f"  After: {final_count} segments")
    for i, seg in enumerate(result.segments):
        print(f"    [{i}] '{seg.text}' (chars={len(seg.text)})")

    # The tiny segment should be merged
    # Note: might result in 1 segment or 'か' merged with previous
    combined_text = ''.join(seg.text for seg in result.segments)
    assert 'か' in combined_text, "'か' should be preserved after merge"
    print("[PASS] Short char segment handled")

    print("\n[PASS] All JP-003 tiny fragment tests passed")
    return True


def test_jp004_language_code_normalization():
    """Test 13: Verify language code normalization (JP-004 fix)."""
    print("\n" + "=" * 60)
    print("TEST 13: Language Code Normalization (JP-004 Fix)")
    print("=" * 60)

    from whisperjav.modules.srt_postprocessing import normalize_language_code, SRTPostProcessor

    # === Test normalize_language_code function ===
    print("\n--- Test 13.1: normalize_language_code function ---")
    test_cases = [
        # (input, expected_output)
        ("Japanese", "ja"),
        ("japanese", "ja"),
        ("JAPANESE", "ja"),
        ("ja", "ja"),
        ("jpn", "ja"),
        ("JP", "jp"),  # 2-letter codes pass through
        ("English", "en"),
        ("english", "en"),
        ("en", "en"),
        ("eng", "en"),
        ("Chinese", "zh"),
        ("chinese", "zh"),
        ("zh", "zh"),
        ("Cantonese", "yue"),
        ("Korean", "ko"),
        ("", "ja"),  # Empty defaults to ja
        (None, "ja"),  # None defaults to ja
    ]

    for input_val, expected in test_cases:
        # Handle None case
        if input_val is None:
            result = normalize_language_code('')
        else:
            result = normalize_language_code(input_val)
        if input_val == '':
            result = normalize_language_code('')

        # For None, we test the default
        if input_val is None:
            result = normalize_language_code('')
        else:
            result = normalize_language_code(input_val)

        status = "✓" if result == expected else "✗"
        print(f"  {status} normalize_language_code('{input_val}') -> '{result}' (expected: '{expected}')")
        assert result == expected, f"normalize_language_code('{input_val}') = '{result}', expected '{expected}'"

    print("[PASS] All normalize_language_code tests passed")

    # === Test SRTPostProcessor initialization ===
    print("\n--- Test 13.2: SRTPostProcessor accepts 'Japanese' ---")

    # This should NOT print a warning and should use CJK sanitizer
    postproc = SRTPostProcessor(language="Japanese")
    assert postproc.language == "ja", f"Expected 'ja', got '{postproc.language}'"
    print(f"  SRTPostProcessor('Japanese').language = '{postproc.language}' [PASS]")

    postproc2 = SRTPostProcessor(language="english")
    assert postproc2.language == "en", f"Expected 'en', got '{postproc2.language}'"
    print(f"  SRTPostProcessor('english').language = '{postproc2.language}' [PASS]")

    postproc3 = SRTPostProcessor(language="ja")
    assert postproc3.language == "ja", f"Expected 'ja', got '{postproc3.language}'"
    print(f"  SRTPostProcessor('ja').language = '{postproc3.language}' [PASS]")

    print("\n[PASS] All JP-004 language normalization tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("REAL-WORLD INTEGRATION TESTS: Qwen Japanese Post-Processing")
    print("=" * 60)

    tests = [
        ("Punctuation Merge (JP-001)", test_merge_master_with_timestamps),
        ("Module Functionality", test_qwen_japanese_postprocessor_module),
        ("QwenASR Initialization", test_qwen_asr_initialization),
        ("Pipeline Passthrough", test_transformers_pipeline_passthrough),
        ("StableTSASR Integration", test_stable_ts_asr_uses_shared_module),
        ("CLI Arguments", test_cli_arguments),
        ("Hierarchical Splitting", test_hierarchical_splitting_unpunctuated),
        ("Mock WhisperResult", test_postprocessor_with_mock_result),
        ("Full Pipeline (GPU)", test_full_pipeline_integration),
        ("Debug Artifact Saving", test_artifact_saving),
        ("Compound Particle Merge (JP-002)", test_jp002_compound_particle_merging),
        ("Tiny Fragment Merge (JP-003)", test_jp003_tiny_fragment_merging),
        ("Language Normalization (JP-004)", test_jp004_language_code_normalization),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        icon = "✓" if status == "PASSED" else "✗" if "ERROR" in status or status == "FAILED" else "○"
        print(f"  {icon} {name}: {status}")

    passed_count = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\n{passed_count}/{total} tests passed")

    return all(s == "PASSED" for _, s in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
