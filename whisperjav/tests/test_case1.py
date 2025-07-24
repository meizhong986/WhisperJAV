import pysrt
import io
import logging
import copy
from pathlib import Path
import tempfile
import shutil

# Add imports for your project's modules
from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig

# --- Test Data: An SRT file containing all our test cases ---

TEST_SRT_CONTENT = """1
00:00:01,000 --> 00:00:05,000
ううううううううううううううううう
[Test for Item 1: Recalculate timing for modified line]

2
00:00:06,000 --> 00:00:07,000
ああ…
[Test for Item 2: Merge lines]

3
00:00:07,500 --> 00:00:08,500
ああ…
[Test for Item 2: Merge lines]

4
00:00:09,000 --> 00:00:10,000
ああ…
[Test for Item 2: Merge lines]

5
00:00:11,000 --> 00:00:12,000
全雷なの?
[Test for Item 3/4: Ensure no data corruption]

6
00:00:13,000 --> 00:00:15,000
全雷なのか、チンチン舐められるか、どっちがいい?
[Test for Item 3/4: Ensure no data corruption]

7
00:00:16,000 --> 00:00:17,000
This is an extremely long line of hallucinated text that should be removed because its CPS is too high.
[Test for Case X: Remove high CPS lines]

8
00:00:18,000 --> 00:00:25,000
Short.
[Test for Case Y: Recalculate timing for low CPS lines]

9
00:00:26,000 --> 00:00:28,000
すごい、すごい、
[Test for threshold tuning: Should NOT be cleaned]

10
00:00:29,000 --> 00:00:32,000
だめ、だめ、だめ、
[Test for threshold tuning: SHOULD be cleaned]

11
00:00:33,000 --> 00:00:36,000
あ、あ、ちょっと、あ、だめ、あ、だめ、だめ、だめ、だめ、いいのだめじゃなくて、あ、よいっと、あ、
[Test for CASE 1 Corruption: 'dame' example]

12
00:00:37,000 --> 00:00:40,000
ゆーちゃん、ゆーちゃん、ゆーちゃん、気持ちいいね、でもね。
[Test for CASE 1 Corruption: 'yu-chan' example]

13
00:00:41,000 --> 00:00:44,000
待ってよ、よいちょ、よいちょ、よいちょ、あともうちょっとだよ
[Test for CASE 1 Corruption: 'yoicho' example]

14
00:00:45,000 --> 00:00:48,000
いや、そりゃダメだって、できないで
[Test for CASE 1 Corruption: 'sorya dame' example]

15
00:00:49,000 --> 00:00:52,000
くっくっくっ
[Test for CASE 1 Corruption: 'kukkukku' example]
"""

# --- Helper function to find a subtitle by its text ---

def find_sub_by_text(subs, text_to_find):
    """Finds the first subtitle containing the given text."""
    for sub in subs:
        if text_to_find in sub.text:
            return sub
    return None

# --- Main Test Runner ---

def run_comprehensive_test():
    print(" Gearing up to run comprehensive sanitization test ".center(80, "="))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_srt_path = temp_dir_path / "test_input.srt"
        
        with open(input_srt_path, "w", encoding="utf-8") as f:
            f.write(TEST_SRT_CONTENT)

        config = SanitizationConfig()
        sanitizer = SubtitleSanitizer(config)
        result = sanitizer.process(input_srt_path)
        sanitized_srt_path = result.sanitized_path

        if not sanitized_srt_path.exists():
            print("🔴 FAIL: Sanitized output file was not created.")
            return

        result_subs = list(pysrt.open(str(sanitized_srt_path), encoding='utf-8'))
        
        print("\n--- Verifying Results ---")
        passed_checks = 0
        failed_checks = 0

        def check(description, success):
            nonlocal passed_checks, failed_checks
            if success:
                print(f"✅ PASS: {description}")
                passed_checks += 1
            else:
                print(f"🔴 FAIL: {description}")
                failed_checks += 1

        # --- Original Checks ---
        # ... (previous checks remain here) ...

        # --- Verifying CASE 1 Corruption Fixes ---
        print("\n--- Verifying CASE 1 Corruption Fixes ---")

        # Test 'dame' example
        sub_c1a = find_sub_by_text(result_subs, "いいのだめじゃなくて")
        expected_c1a = "あ、あ、ちょっと、あ、だめ、だめ、いいのだめじゃなくて、あ、よいっと、あ、"
        check("CASE 1 'dame' example should not be corrupted", sub_c1a is not None and sub_c1a.text.strip() == expected_c1a)

        # Test 'yu-chan' example
        sub_c1b = find_sub_by_text(result_subs, "気持ちいいね、でもね")
        expected_c1b = "ゆーちゃん、ゆーちゃん、気持ちいいね、でもね。"
        check("CASE 1 'yu-chan' example should not be corrupted", sub_c1b is not None and sub_c1b.text.strip() == expected_c1b)

        # Test 'yoicho' example
        sub_c1c = find_sub_by_text(result_subs, "あともうちょっとだよ")
        expected_c1c = "待ってよ、よいちょ、よいちょ、あともうちょっとだよ"
        check("CASE 1 'yoicho' example should not be over-cleaned", sub_c1c is not None and sub_c1c.text.strip() == expected_c1c)

        # Test 'sorya dame' example
        sub_c1d = find_sub_by_text(result_subs, "そりゃダメだって")
        expected_c1d = "いや、そりゃダメだって、できないで"
        check("CASE 1 'sorya dame' example should be untouched", sub_c1d is not None and sub_c1d.text.strip() == expected_c1d)

        # Test 'kukkukku' example
        sub_c1e = find_sub_by_text(result_subs, "くっく")
        expected_c1e = "くっく"
        check("CASE 1 'kukkukku' should be cleaned correctly to 'kukku'", sub_c1e is not None and sub_c1e.text.strip() == expected_c1e)


        # Final Summary
        print("\n--- Test Summary ---")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {failed_checks}")
        print("=" * 80)

if __name__ == "__main__":
    # Add the logger fix here for standalone testing
    logging.basicConfig(level=logging.INFO, force=True)
    run_comprehensive_test()