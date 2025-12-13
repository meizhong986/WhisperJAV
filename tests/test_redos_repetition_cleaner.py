"""
Test case to verify ReDoS (Regular Expression Denial of Service) vulnerability
in repetition_cleaner.py

This test demonstrates that certain regex patterns with nested quantifiers
can cause catastrophic backtracking when processing long input strings.

Run with: python tests/test_redos_repetition_cleaner.py

Expected result: Processing time should grow EXPONENTIALLY with input length
for vulnerable patterns, while safe patterns should grow LINEARLY.
"""

import regex
import time
import signal
import sys
from typing import Tuple, Optional
from contextlib import contextmanager

# Timeout handling for Windows compatibility
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds: float):
    """Context manager to limit execution time (cross-platform)."""
    import threading

    result = {'completed': False, 'exception': None}

    def target(func_result):
        try:
            func_result['value'] = yield
            func_result['completed'] = True
        except Exception as e:
            func_result['exception'] = e

    # For simplicity, we'll just measure time and report if it exceeds limit
    # True timeout would require multiprocessing
    yield


def measure_regex_time(pattern: regex.Pattern, text: str, timeout_sec: float = 10.0) -> Tuple[float, bool, Optional[str]]:
    """
    Measure how long a regex substitution takes.
    Returns: (time_seconds, timed_out, result_or_none)
    """
    start = time.perf_counter()

    try:
        # Use regex with timeout if available (regex module supports this)
        result = pattern.sub(r'\1', text, timeout=timeout_sec)
        elapsed = time.perf_counter() - start
        return elapsed, False, result
    except TimeoutError:
        elapsed = time.perf_counter() - start
        return elapsed, True, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return elapsed, False, f"ERROR: {e}"


def generate_test_strings():
    """Generate test strings that should trigger ReDoS."""

    test_cases = []

    # Case 1: Japanese text with separators (NO repetition - worst case for backtracking)
    # This is the dangerous case: text that LOOKS like it might match but doesn't
    for length in [10, 20, 30, 40, 50, 75, 100, 150, 200]:
        # Mixed kana with punctuation - no actual repetition pattern
        text = "".join([
            "あいうえお、かきくけこ。さしすせそ！たちつてと？"
        ] * (length // 20 + 1))[:length]
        test_cases.append(('no_repetition_mixed', length, text))

    # Case 2: Long string of single characters with separators (should match)
    for length in [10, 20, 30, 40, 50, 75, 100, 150, 200]:
        text = "あ、" * (length // 2)
        test_cases.append(('has_repetition', length, text))

    # Case 3: Realistic subtitle - long dialogue without repetition
    for length in [50, 100, 200, 300, 400, 500]:
        base = "それはとても難しい問題ですね、私たちはどうすればいいのでしょうか、本当に困っています"
        text = (base * (length // len(base) + 1))[:length]
        test_cases.append(('realistic_dialogue', length, text))

    # Case 4: Worst case - alternating pattern that almost matches
    for length in [20, 30, 40, 50, 60, 70, 80]:
        # This creates text like "あ、い、う、え、お、" which the pattern
        # tries many ways to match before failing
        chars = "あいうえおかきくけこさしすせそたちつてと"
        text = "、".join(chars[i % len(chars)] for i in range(length))
        test_cases.append(('alternating_worst', length, text))

    return test_cases


def test_pattern(name: str, pattern: regex.Pattern, test_cases: list, timeout: float = 5.0):
    """Test a single pattern against all test cases."""

    print(f"\n{'='*70}")
    print(f"PATTERN: {name}")
    print(f"Regex: {pattern.pattern[:60]}...")
    print(f"{'='*70}")
    print(f"{'Type':<25} {'Length':>8} {'Time (ms)':>12} {'Status':<15}")
    print("-" * 70)

    results = []

    for case_type, length, text in test_cases:
        elapsed, timed_out, result = measure_regex_time(pattern, text, timeout)
        elapsed_ms = elapsed * 1000

        if timed_out:
            status = "TIMEOUT!"
            flag = "***"
        elif elapsed_ms > 1000:
            status = "VERY SLOW"
            flag = "**"
        elif elapsed_ms > 100:
            status = "SLOW"
            flag = "*"
        else:
            status = "OK"
            flag = ""

        print(f"{case_type:<25} {length:>8} {elapsed_ms:>12.2f} {status:<15} {flag}")
        results.append((case_type, length, elapsed_ms, timed_out))

    return results


def analyze_exponential_growth(results: list):
    """Check if timing shows exponential growth."""

    # Group by case type
    by_type = {}
    for case_type, length, time_ms, timed_out in results:
        if case_type not in by_type:
            by_type[case_type] = []
        by_type[case_type].append((length, time_ms, timed_out))

    print("\n" + "="*70)
    print("GROWTH ANALYSIS")
    print("="*70)

    for case_type, data in by_type.items():
        data.sort(key=lambda x: x[0])

        if len(data) < 2:
            continue

        # Check growth rate between consecutive measurements
        growth_rates = []
        for i in range(1, len(data)):
            len_prev, time_prev, _ = data[i-1]
            len_curr, time_curr, _ = data[i]

            if time_prev > 0.1:  # Only analyze if previous time was measurable
                len_ratio = len_curr / len_prev
                time_ratio = time_curr / time_prev
                growth_rates.append((len_ratio, time_ratio))

        if growth_rates:
            avg_len_ratio = sum(r[0] for r in growth_rates) / len(growth_rates)
            avg_time_ratio = sum(r[1] for r in growth_rates) / len(growth_rates)

            # Exponential: time_ratio >> len_ratio
            # Linear: time_ratio ≈ len_ratio
            if avg_time_ratio > avg_len_ratio * 2:
                growth_type = "EXPONENTIAL (ReDoS likely!)"
            elif avg_time_ratio > avg_len_ratio * 1.5:
                growth_type = "SUPER-LINEAR (potential issue)"
            else:
                growth_type = "LINEAR (safe)"

            print(f"\n{case_type}:")
            print(f"  Avg length growth: {avg_len_ratio:.2f}x")
            print(f"  Avg time growth:   {avg_time_ratio:.2f}x")
            print(f"  Assessment:        {growth_type}")


def main():
    print("="*70)
    print("ReDoS (Regular Expression Denial of Service) Test")
    print("Testing repetition_cleaner.py patterns")
    print("="*70)

    # The problematic patterns from repetition_cleaner.py
    patterns = {
        'phrase_with_separator': regex.compile(
            r'((?:[\p{L}\p{N}]{1,8}[、,!\s!!??。。・]+))\1{3,}'
        ),
        'phrase_with_comma': regex.compile(
            r'((?:[\p{L}\p{N}]{1,10}[、,]\s*))\1{2,}'
        ),
        'multi_char_word': regex.compile(
            r'(([ぁ-んァ-ン]{2,4}))\1{3,}'
        ),
        'single_char_flood': regex.compile(
            r'([ぁ-んァ-ン])\1{3,}'
        ),
    }

    # Safe comparison pattern (no nested quantifiers)
    patterns['safe_simple'] = regex.compile(r'(.)\1{3,}')

    test_cases = generate_test_strings()
    all_results = {}

    for name, pattern in patterns.items():
        results = test_pattern(name, pattern, test_cases, timeout=5.0)
        all_results[name] = results

    # Analyze growth patterns
    print("\n\n")
    print("="*70)
    print("SUMMARY: WHICH PATTERNS SHOW EXPONENTIAL GROWTH?")
    print("="*70)

    for name, results in all_results.items():
        print(f"\n--- {name} ---")
        analyze_exponential_growth(results)

    print("\n\n")
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If you see EXPONENTIAL growth for 'phrase_with_separator' or 'phrase_with_comma'
patterns on non-repeating input, this confirms the ReDoS vulnerability.

The crash in Pass 2 is likely caused by:
1. FasterWhisperProASR producing longer subtitle chunks (due to aggressive params)
2. One or more subtitles containing 200+ characters of mixed Japanese text
3. The regex engine spending 60+ seconds trying all backtracking combinations
4. Windows killing the subprocess due to timeout or resource exhaustion

RECOMMENDED FIX:
1. Add input length guard (skip regex for text > 200 chars)
2. Rewrite patterns to avoid nested quantifiers
3. Use possessive quantifiers or atomic groups if available
""")


if __name__ == "__main__":
    main()
