"""
Comprehensive ReDoS test for ALL patterns in repetition_cleaner.py

Tests both the cleaning patterns AND the _is_all_repetition patterns.
"""

import regex
import time
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def time_regex_match(pattern, text, max_seconds=30):
    """Time a regex match with timeout."""
    start = time.perf_counter()
    try:
        result = pattern.match(text, timeout=max_seconds)
        elapsed = time.perf_counter() - start
        return elapsed, False, result is not None
    except TimeoutError:
        return max_seconds, True, None


def time_regex_sub(pattern, text, max_seconds=30):
    """Time a regex substitution with timeout."""
    start = time.perf_counter()
    try:
        result = pattern.sub(r'\1', text, timeout=max_seconds)
        elapsed = time.perf_counter() - start
        return elapsed, False, result
    except TimeoutError:
        return max_seconds, True, None


def test_pattern(name, pattern, test_func, test_strings, max_time=30):
    """Test a pattern with various strings."""
    print(f"\n{'='*70}")
    print(f"PATTERN: {name}")
    print(f"{'='*70}")

    results = []

    for desc, text in test_strings:
        length = len(text)
        elapsed, timed_out, result = test_func(pattern, text, max_time)
        elapsed_ms = elapsed * 1000

        if timed_out:
            status = "TIMEOUT!"
        elif elapsed_ms > 1000:
            status = "VERY SLOW"
        elif elapsed_ms > 100:
            status = "SLOW"
        else:
            status = "ok"

        print(f"  {desc:<30} len={length:>4}  time={elapsed_ms:>10.2f}ms  {status}")
        results.append((desc, length, elapsed_ms, timed_out))

        if timed_out:
            print("    >>> TIMEOUT - This pattern causes ReDoS!")
            break

    return results


def main():
    print("="*70)
    print("COMPREHENSIVE ReDoS TEST FOR repetition_cleaner.py")
    print("="*70)

    # All patterns from repetition_cleaner.py

    # 1. Cleaning patterns (line 53-97)
    cleaning_patterns = {
        'phrase_with_separator': regex.compile(
            r'((?:[\p{L}\p{N}]{1,8}[、,!\s!!??。。・]+))\1{3,}'
        ),
        'multi_char_word': regex.compile(
            r'(([ぁ-んァ-ン]{2,4}))\1{3,}'
        ),
        'phrase_with_comma': regex.compile(
            r'((?:[\p{L}\p{N}]{1,10}[、,]\s*))\1{2,}'
        ),
        'single_char_whitespace_flood': regex.compile(
            r'([ぁ-んァ-ン])(?:[\s　]*\1){3,}'
        ),
        'prefix_plus_char': regex.compile(
            r'([ぁ-んァ-ン]{1,2})([ぁ-んァ-ン])\2{3,}'
        ),
        'single_char_flood': regex.compile(
            r'([ぁ-んァ-ン])\1{3,}'
        ),
    }

    # 2. Patterns from _is_all_repetition (line 197-206) - MOST DANGEROUS
    validation_patterns = {
        'is_all_rep_1_DANGEROUS': regex.compile(
            r'^((?:.{1,5}?)[、,!\s!?・]){5,}$'
        ),
        'is_all_rep_2_DANGEROUS': regex.compile(
            r'^((?:.{2,5}?))\1{3,}$'
        ),
    }

    # Test strings designed to trigger backtracking
    test_strings_sub = [
        # Normal text - should be fast
        ("short_simple", "あいうえお"),
        ("medium_simple", "あいうえお" * 10),

        # Text with separators but no repetition
        ("separators_20", "あ、い！う。え？お、か！き。く？け、" * 2),
        ("separators_50", "あ、い！う。え？お、" * 5),
        ("separators_100", "あ、い！う。え？お、" * 10),
        ("separators_200", "あ、い！う。え？お、" * 20),

        # Long dialogue-like text
        ("dialogue_50", "それは本当に難しい問題ですね" * 2),
        ("dialogue_100", "それは本当に難しい問題ですね" * 5),
        ("dialogue_200", "それは本当に難しい問題ですね" * 10),

        # Mixed content - realistic subtitle
        ("realistic_100", "ああ、そうだね。うん、分かった。えーと、それで？" * 2),
        ("realistic_200", "ああ、そうだね。うん、分かった。えーと、それで？" * 4),
        ("realistic_300", "ああ、そうだね。うん、分かった。えーと、それで？" * 6),
    ]

    # Test strings for match patterns (is_all_repetition)
    test_strings_match = [
        ("short_5", "あ、い、う、え、お、"),
        ("medium_20", "あ、い、う、え、お、か、き、く、け、こ、" * 2),
        ("long_50", "あ、" * 25),
        ("long_100", "あ、" * 50),
        ("varied_50", "".join(f"{c}、" for c in "あいうえおかきくけこ" * 5)),
        ("varied_100", "".join(f"{c}、" for c in "あいうえおかきくけこ" * 10)),
        ("varied_150", "".join(f"{c}、" for c in "あいうえおかきくけこ" * 15)),
    ]

    # Test cleaning patterns (use sub)
    print("\n" + "="*70)
    print("TESTING CLEANING PATTERNS (pattern.sub)")
    print("="*70)

    for name, pattern in cleaning_patterns.items():
        test_pattern(name, pattern, time_regex_sub, test_strings_sub)

    # Test validation patterns (use match) - MOST LIKELY TO CAUSE ReDoS
    print("\n" + "="*70)
    print("TESTING VALIDATION PATTERNS (pattern.match) - _is_all_repetition")
    print("These are the MOST DANGEROUS patterns!")
    print("="*70)

    for name, pattern in validation_patterns.items():
        test_pattern(name, pattern, time_regex_match, test_strings_match)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
KEY FINDINGS:

1. If any pattern shows TIMEOUT or time > 1000ms, it confirms ReDoS.

2. The patterns in _is_all_repetition() are the MOST DANGEROUS:
   - They use .{1,5}? which matches ANY character
   - Combined with {5,}$ or \1{3,}$, this creates massive backtracking

3. If tests pass quickly, the `regex` module may have optimizations
   that prevent worst-case backtracking. However, edge cases in
   real subtitle data may still trigger the issue.

NEXT STEPS:
- If ReDoS confirmed: Implement fix
- If not confirmed here: Need to examine actual subtitle data that crashed
""")


if __name__ == "__main__":
    main()
