import pysrt
import io
import logging
import copy
from pathlib import Path
import tempfile
import shutil
import regex

# Add imports for your project's modules
from whisperjav.modules.repetition_cleaner import RepetitionCleaner
from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig
from whisperjav.config.sanitization_constants import RepetitionConstants

def run_pattern_diagnostics(cleaner):
    """Applies each pattern individually to see which one causes the error."""
    print(" Pattern-by-Pattern Diagnostic Report ".center(80, "#"))
    
    problem_lines = [
        "ゆーちゃん、ゆーちゃん、ゆーちゃん、気持ちいいね、でもね。",
        "あ、あ、ちょっと、あ、だめ、あ、だめ、だめ、だめ、だめ、いいのだめじゃなくて、あ、よいっと、あ、",
        "いや、そりゃダメだって、できないで",
        "くっくっくっ"
    ]

    # CORRECTED: The attribute is now `cleaner.cleaning_patterns`
    # The loop now unpacks the 'name' of the pattern as well.
    for i, (name, pattern, replacement) in enumerate(cleaner.cleaning_patterns):
        print(f"\n--- Testing Pattern: '{name}' ---")
        print(f"Regex: {pattern.pattern}")
        
        for line in problem_lines:
            try:
                result = pattern.sub(replacement, line)
                if result != line:
                    print(f"  ✅ TRIGGERED on input: '{line[:40]}...'")
                    print(f"     Output: '{result}'")
            except Exception as e:
                print(f"  ❌ FAILED on input '{line[:40]}...' with error: {e}")

    print("\n" + "#" * 80)


def run_comprehensive_test():
    """Main test runner."""
    print(" Gearing up for final diagnostic... ".center(80, "="))
    
    # 1. Initialize the Sanitizer
    config = SanitizationConfig()
    constants = config.get_effective_constants()
    cleaner = RepetitionCleaner(constants['repetition'])

    # !!!!!!!!!!!!! RUN DIAGNOSTICS !!!!!!!!!!!!!
    run_pattern_diagnostics(cleaner)
    
    print("\nNOTE: Diagnostic complete.")
    print("Please review the report above to see which patterns trigger and what their output is.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run_comprehensive_test()