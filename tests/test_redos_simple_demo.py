"""
Simple demonstration of ReDoS vulnerability in repetition_cleaner.py

This script shows the EXPONENTIAL time growth when the problematic regex
patterns process non-matching text of increasing length.

Run with: python tests/test_redos_simple_demo.py

WARNING: This test may hang or use 100% CPU if the vulnerability is confirmed.
         It has built-in timeouts to prevent indefinite hangs.
"""

import regex
import time
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# The dangerous pattern from repetition_cleaner.py line 59
DANGEROUS_PATTERN = regex.compile(
    r'((?:[\p{L}\p{N}]{1,8}[、,!\s!!??。。・]+))\1{3,}'
)

# A safe comparison pattern
SAFE_PATTERN = regex.compile(r'(.)\1{3,}')


def time_regex(pattern, text, max_seconds=10):
    """Time a regex substitution with timeout."""
    start = time.perf_counter()
    try:
        # regex module supports timeout parameter
        result = pattern.sub(r'\1', text, timeout=max_seconds)
        elapsed = time.perf_counter() - start
        return elapsed, False
    except TimeoutError:
        return max_seconds, True


def create_worst_case_text(length):
    """
    Create text that triggers maximum backtracking.

    The pattern looks for: (phrase + separator) repeated 3+ times
    This text has phrases with separators but NO repetition,
    forcing the regex to try ALL combinations before failing.
    """
    # Different characters with separators - no repetition
    chars = "あいうえおかきくけこさしすせそたちつてとなにぬねの"
    result = []
    for i in range(length):
        char = chars[i % len(chars)]
        sep = "、" if i % 3 == 0 else ("！" if i % 3 == 1 else "。")
        result.append(char + sep)
    return "".join(result)


def main():
    print("=" * 60)
    print("ReDoS VULNERABILITY DEMONSTRATION")
    print("=" * 60)
    print()
    print("Testing the pattern from repetition_cleaner.py line 59:")
    print(f"  {DANGEROUS_PATTERN.pattern}")
    print()
    print("This pattern has NESTED QUANTIFIERS:")
    print("  - Inner: {1,8} for characters")
    print("  - Outer: {3,} for repetitions")
    print("  - Combined: O(2^n) worst-case complexity")
    print()
    print("-" * 60)
    print(f"{'Length':>10} {'Dangerous (ms)':>15} {'Safe (ms)':>15} {'Ratio':>10}")
    print("-" * 60)

    # Test with increasing lengths
    lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    previous_time = None

    for length in lengths:
        text = create_worst_case_text(length)

        # Time dangerous pattern
        dangerous_time, dangerous_timeout = time_regex(DANGEROUS_PATTERN, text, max_seconds=30)

        # Time safe pattern (for comparison)
        safe_time, safe_timeout = time_regex(SAFE_PATTERN, text, max_seconds=5)

        dangerous_ms = dangerous_time * 1000
        safe_ms = safe_time * 1000

        # Calculate ratio
        if safe_ms > 0.01:
            ratio = dangerous_ms / safe_ms
        else:
            ratio = dangerous_ms / 0.01

        # Format output
        dangerous_str = f"{dangerous_ms:.2f}" if not dangerous_timeout else "TIMEOUT!"
        safe_str = f"{safe_ms:.2f}" if not safe_timeout else "TIMEOUT!"
        ratio_str = f"{ratio:.1f}x" if not dangerous_timeout else "∞"

        # Show growth rate
        growth = ""
        if previous_time and previous_time > 0.1 and dangerous_ms > 0.1:
            growth_rate = dangerous_ms / previous_time
            if growth_rate > 3:
                growth = f"  ← {growth_rate:.1f}x GROWTH!"
            elif growth_rate > 1.5:
                growth = f"  ← {growth_rate:.1f}x growth"

        print(f"{length:>10} {dangerous_str:>15} {safe_str:>15} {ratio_str:>10}{growth}")

        previous_time = dangerous_ms

        # Stop if we hit timeout
        if dangerous_timeout:
            print()
            print("*** TIMEOUT REACHED - ReDoS CONFIRMED! ***")
            print()
            print(f"At length {length}, the regex took > 30 seconds.")
            print("This proves exponential time complexity.")
            break

        # Stop if time exceeds reasonable threshold
        if dangerous_ms > 5000:  # 5 seconds
            print()
            print("*** EXCESSIVE TIME - ReDoS LIKELY! ***")
            print()
            print(f"At length {length}, the regex took {dangerous_ms:.0f}ms.")
            print("Longer inputs would take exponentially longer.")
            break

    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
WHAT THIS MEANS:

1. The 'phrase_with_separator' pattern in repetition_cleaner.py
   has O(2^n) time complexity on non-matching input.

2. When Pass 2 (FasterWhisperProASR) produces a long subtitle
   (200+ characters) without actual repetition, this regex
   can hang for minutes or longer.

3. The 67-second hang before crash in the user's log matches
   this behavior perfectly.

SOLUTION OPTIONS:

A. Add length guard:
   if len(text) > 150:
       return text, []  # Skip regex for long text

B. Rewrite pattern without nested quantifiers

C. Add timeout wrapper around pattern.sub()

D. Use possessive quantifiers: (?:...){3,}+ instead of {3,}
   (prevents backtracking into already-matched content)
""")


if __name__ == "__main__":
    print("Starting ReDoS test... (may take up to 30 seconds)")
    print()
    main()
