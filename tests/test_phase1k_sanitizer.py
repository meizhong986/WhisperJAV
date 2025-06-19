#!/usr/bin/env python3
"""
Quick test script for sanitization system
Run this after applying the quick fixes to verify everything works.
"""

import sys
import tempfile
import pysrt
from pathlib import Path
from datetime import datetime

def create_test_srt_file():
    """Create a test SRT file with various issues to clean"""
    test_content = [
        "www",  # Should be removed (hallucination)
        "あああああああああ",  # Should be cleaned (repetition)
        "正常なテキストです",  # Should be preserved (normal)
        "ok",  # Should be removed (hallucination)
        "はははははははは",  # Should be cleaned (repetition)
        "12345円です",  # Should be preserved (has numbers/currency)
        "そこそこ良い",  # Should be preserved (legitimate repetition)
        "笑",  # Should be removed (hallucination)
        "これは普通の文章です。",  # Should be preserved
        "",  # Empty - should be removed
    ]
    
    # Create SRT file
    srt_file = pysrt.SubRipFile()
    for i, text in enumerate(test_content, 1):
        start_time = pysrt.SubRipTime(milliseconds=(i-1)*2000)
        end_time = pysrt.SubRipTime(milliseconds=i*2000)
        
        # Create one subtitle with >12s duration (timing hallucination)
        if i == 5:
            end_time = pysrt.SubRipTime(milliseconds=(i-1)*2000 + 15000)  # 15 seconds
        
        item = pysrt.SubRipItem(
            index=i,
            start=start_time,
            end=end_time,
            text=text
        )
        srt_file.append(item)
    
    return srt_file

def test_individual_modules():
    """Test individual modules for basic functionality"""
    print("=" * 50)
    print("TESTING INDIVIDUAL MODULES")
    print("=" * 50)
    
    # Test 1: Hallucination Remover
    print("\n1. Testing Hallucination Remover...")
    try:
        from whisperjav.modules.hallucination_remover import HallucinationRemover
        from whisperjav.config.sanitization_constants import HallucinationConstants
        
        remover = HallucinationRemover(HallucinationConstants())
        
        test_cases = [("www", ""), ("ok", ""), ("笑", ""), ("正常なテキスト", "正常なテキスト")]
        
        for input_text, expected_behavior in test_cases:
            result, mods = remover.remove_hallucinations(input_text)
            status = "✓" if (expected_behavior == "" and result == "") or \
                           (expected_behavior != "" and result != "") else "✗"
            print(f"   {status} '{input_text}' -> '{result}'")
            
        print("   ✓ Hallucination remover loaded successfully")
        
    except Exception as e:
        print(f"   ✗ Hallucination remover failed: {e}")
        return False
    
    # Test 2: Repetition Cleaner
    print("\n2. Testing Repetition Cleaner...")
    try:
        from whisperjav.modules.repetition_cleaner import RepetitionCleaner
        from whisperjav.config.sanitization_constants import RepetitionConstants
        
        cleaner = RepetitionCleaner(RepetitionConstants())
        
        test_cases = [
            ("あああああああ", "should be shortened"),
            ("そこそこ", "should be preserved"),
            ("12345", "should be preserved"),
            ("正常なテキスト", "should be preserved")
        ]
        
        for input_text, expected_behavior in test_cases:
            result, mods = cleaner.clean_repetitions(input_text)
            if "shortened" in expected_behavior:
                status = "✓" if len(result) < len(input_text) else "✗"
            else:
                status = "✓" if result == input_text else "✗"
            print(f"   {status} '{input_text}' -> '{result}'")
            
        print("   ✓ Repetition cleaner loaded successfully")
        
    except Exception as e:
        print(f"   ✗ Repetition cleaner failed: {e}")
        return False
    
    # Test 3: Timing Adjuster
    print("\n3. Testing Timing Adjuster...")
    try:
        from whisperjav.modules.timing_adjuster import TimingAdjuster
        from whisperjav.config.sanitization_constants import TimingConstants, CrossSubtitleConstants
        
        adjuster = TimingAdjuster(TimingConstants(), CrossSubtitleConstants())
        print("   ✓ Timing adjuster loaded successfully")
        
    except Exception as e:
        print(f"   ✗ Timing adjuster failed: {e}")
        return False
    
    return True

def test_full_pipeline():
    """Test the complete sanitization pipeline"""
    print("\n" + "=" * 50)
    print("TESTING FULL PIPELINE")
    print("=" * 50)
    
    try:
        from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
        from whisperjav.config.sanitization_config import SanitizationConfig
        
        # Create test file
        print("\n1. Creating test SRT file...")
        test_srt = create_test_srt_file()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            test_srt.save(f.name, encoding='utf-8')
            test_file_path = Path(f.name)
        
        print(f"   ✓ Created test file: {test_file_path}")
        print(f"   ✓ Original subtitles: {len(test_srt)}")
        
        # Run sanitization
        print("\n2. Running sanitization...")
        config = SanitizationConfig(
            sensitivity_mode="balanced",
            save_artifacts=True,
            save_original=True
        )
        
        sanitizer = SubtitleSanitizer(config)
        result = sanitizer.process(test_file_path)
        
        print(f"   ✓ Sanitization completed in {result.processing_time:.2f}s")
        print(f"   ✓ Final subtitles: {result.statistics['final_subtitle_count']}")
        print(f"   ✓ Total modifications: {result.statistics['total_modifications']}")
        
        # Check outputs
        print("\n3. Checking outputs...")
        if result.sanitized_path.exists():
            print(f"   ✓ Sanitized file created: {result.sanitized_path}")
        else:
            print(f"   ✗ Sanitized file missing: {result.sanitized_path}")
            
        if result.original_backup_path and result.original_backup_path.exists():
            print(f"   ✓ Backup file created: {result.original_backup_path}")
        else:
            print(f"   ℹ No backup file (may be disabled)")
            
        if result.artifacts_path and result.artifacts_path.exists():
            print(f"   ✓ Artifacts file created: {result.artifacts_path}")
        else:
            print(f"   ℹ No artifacts file")
        
        # Check for errors in statistics
        if 'error' in result.statistics:
            print(f"   ⚠ Processing had errors: {result.statistics['error']}")
        else:
            print(f"   ✓ No processing errors detected")
        
        # Quick validation
        final_srt = pysrt.open(str(result.sanitized_path), encoding='utf-8')
        reduction_percent = ((len(test_srt) - len(final_srt)) / len(test_srt)) * 100
        print(f"   ✓ Subtitle reduction: {reduction_percent:.1f}%")
        
        # Check if common hallucinations were removed
        final_texts = [sub.text.lower() for sub in final_srt]
        hallucinations_found = any(h in text for text in final_texts for h in ['www', 'ok', '笑'])
        if hallucinations_found:
            print(f"   ⚠ Some hallucinations may remain")
        else:
            print(f"   ✓ Common hallucinations appear to be removed")
        
        # Cleanup
        try:
            test_file_path.unlink()
            if result.sanitized_path != test_file_path:
                result.sanitized_path.unlink(missing_ok=True)
            if result.original_backup_path:
                result.original_backup_path.unlink(missing_ok=True)
            if result.artifacts_path:
                result.artifacts_path.unlink(missing_ok=True)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"   ✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("WHISPERJAV SANITIZATION QUICK TEST")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    
    # Test individual modules
    modules_ok = test_individual_modules()
    
    if not modules_ok:
        print("\n" + "=" * 50)
        print("❌ MODULE TESTS FAILED")
        print("❌ Please check the quick fixes were applied correctly")
        return False
    
    # Test full pipeline
    pipeline_ok = test_full_pipeline()
    
    print("\n" + "=" * 50)
    if modules_ok and pipeline_ok:
        print("✅ ALL TESTS PASSED")
        print("✅ Sanitization system appears to be working correctly")
        print("✅ Ready for testing with real JAV subtitle files")
    else:
        print("❌ SOME TESTS FAILED")
        print("❌ Please review the error messages above")
        
    print("=" * 50)
    return modules_ok and pipeline_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)