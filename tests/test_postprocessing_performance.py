"""
Test the actual post-processing flow to identify performance bottlenecks.

This test simulates what happens during Step 5 (Post-processing subtitles)
to identify where the 67-second hang occurs.
"""

import time
import sys
import io
import tempfile
import pysrt
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_srt(num_subtitles: int, avg_text_length: int = 30) -> Path:
    """Create a test SRT file with specified number of subtitles."""

    # Sample Japanese text patterns (realistic dialogue)
    text_samples = [
        "ああ、そうだね。",
        "うん、分かった。",
        "えーと、それで？",
        "本当に？信じられない。",
        "どうしようかな…",
        "ねえ、聞いてる？",
        "はい、もちろんです。",
        "いや、違うと思う。",
        "そんなことないよ。",
        "まあ、いいけど。",
        "ちょっと待って。",
        "何それ？",
        "やめてよ。",
        "すごいね！",
        "大丈夫？",
        # Longer samples
        "それは本当に難しい問題ですね、私たちはどうすればいいのでしょうか。",
        "ああ、あの人のことね。最近全然会ってないんだよね。どうしてるかな。",
        "えっと、その件については後で改めて連絡させていただきます。",
        # Very long sample (potential problem case)
        "ああああああああああああああああああああああああああああああ",
        "うんうんうんうんうんうんうんうんうんうんうんうんうんうんうん",
    ]

    subs = []
    current_time = 0

    for i in range(num_subtitles):
        text = text_samples[i % len(text_samples)]

        # Make some subtitles longer
        if i % 50 == 0:
            text = text * 3  # Triple length every 50 subs

        start = pysrt.SubRipTime(milliseconds=current_time)
        duration = max(1000, len(text) * 100)  # ~100ms per character
        end = pysrt.SubRipTime(milliseconds=current_time + duration)

        sub = pysrt.SubRipItem(index=i+1, start=start, end=end, text=text)
        subs.append(sub)

        current_time += duration + 500  # 500ms gap

    # Create temp file
    temp_path = Path(tempfile.mktemp(suffix='.srt'))
    pysrt.SubRipFile(subs).save(str(temp_path), encoding='utf-8')

    return temp_path


def test_component(name: str, func, *args, **kwargs):
    """Time a component and report."""
    print(f"  Testing {name}...", end=" ", flush=True)
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{elapsed*1000:.2f}ms")
        return elapsed, result
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"ERROR after {elapsed*1000:.2f}ms: {e}")
        return elapsed, None


def test_full_postprocessing():
    """Test the full post-processing pipeline."""

    print("="*70)
    print("POST-PROCESSING PERFORMANCE TEST")
    print("="*70)

    # Test with increasing subtitle counts
    test_sizes = [50, 100, 200, 500]

    for num_subs in test_sizes:
        print(f"\n--- Testing with {num_subs} subtitles ---")

        # Create test file
        print("  Creating test SRT...", end=" ", flush=True)
        start = time.perf_counter()
        srt_path = create_test_srt(num_subs)
        print(f"{(time.perf_counter() - start)*1000:.2f}ms")

        try:
            # Test individual components

            # 1. Load SRT
            elapsed_load, subs = test_component(
                "pysrt.open()",
                lambda: list(pysrt.open(str(srt_path), encoding='utf-8'))
            )

            if subs is None:
                continue

            # 2. Import and create sanitizer
            print("  Importing SubtitleSanitizer...", end=" ", flush=True)
            start = time.perf_counter()
            from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
            from whisperjav.config.sanitization_config import SanitizationConfig
            config = SanitizationConfig()
            print(f"{(time.perf_counter() - start)*1000:.2f}ms")

            # 3. Create sanitizer instance (includes loading hallucination patterns)
            elapsed_init, sanitizer = test_component(
                "SubtitleSanitizer.__init__()",
                lambda: SubtitleSanitizer(config)
            )

            if sanitizer is None:
                continue

            # 4. Test HallucinationRemover separately
            print("  Testing HallucinationRemover on each subtitle...")
            start = time.perf_counter()
            hall_times = []
            for sub in subs[:min(50, len(subs))]:  # Test first 50
                sub_start = time.perf_counter()
                sanitizer.hallucination_remover.remove_hallucinations(sub.text, 'ja')
                hall_times.append(time.perf_counter() - sub_start)
            elapsed_hall = time.perf_counter() - start
            avg_hall = sum(hall_times) / len(hall_times) * 1000 if hall_times else 0
            max_hall = max(hall_times) * 1000 if hall_times else 0
            print(f"    Total: {elapsed_hall*1000:.2f}ms, Avg per sub: {avg_hall:.3f}ms, Max: {max_hall:.3f}ms")

            # 5. Test RepetitionCleaner separately
            print("  Testing RepetitionCleaner on each subtitle...")
            start = time.perf_counter()
            rep_times = []
            for sub in subs[:min(50, len(subs))]:  # Test first 50
                sub_start = time.perf_counter()
                sanitizer.repetition_cleaner.clean_repetitions(sub.text)
                rep_times.append(time.perf_counter() - sub_start)
            elapsed_rep = time.perf_counter() - start
            avg_rep = sum(rep_times) / len(rep_times) * 1000 if rep_times else 0
            max_rep = max(rep_times) * 1000 if rep_times else 0
            print(f"    Total: {elapsed_rep*1000:.2f}ms, Avg per sub: {avg_rep:.3f}ms, Max: {max_rep:.3f}ms")

            # 6. Test full sanitizer.process()
            print("  Testing full sanitizer.process()...", end=" ", flush=True)
            start = time.perf_counter()
            try:
                result = sanitizer.process(srt_path)
                elapsed_full = time.perf_counter() - start
                print(f"{elapsed_full*1000:.2f}ms")
                print(f"    Result: {result.statistics.get('original_subtitle_count', '?')} -> {result.statistics.get('final_subtitle_count', '?')} subtitles")
            except Exception as e:
                elapsed_full = time.perf_counter() - start
                print(f"ERROR after {elapsed_full*1000:.2f}ms")
                print(f"    Error: {e}")

        finally:
            # Cleanup
            try:
                srt_path.unlink()
            except:
                pass

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


def test_long_subtitle_edge_case():
    """Test with extremely long subtitles that might cause issues."""

    print("\n" + "="*70)
    print("EDGE CASE: EXTREMELY LONG SUBTITLES")
    print("="*70)

    from whisperjav.modules.repetition_cleaner import RepetitionCleaner
    from whisperjav.config.sanitization_constants import RepetitionConstants

    cleaner = RepetitionCleaner(RepetitionConstants())

    # Test cases that might be problematic
    test_cases = [
        ("normal_50", "これは普通のテキストです。" * 5),
        ("normal_100", "これは普通のテキストです。" * 10),
        ("repetitive_50", "あああああ" * 10),
        ("repetitive_100", "あああああ" * 20),
        ("mixed_punct_100", "あ、い！う。え？お、" * 10),
        ("mixed_punct_200", "あ、い！う。え？お、" * 20),
        ("mixed_punct_500", "あ、い！う。え？お、" * 50),
        ("long_no_repeat_500", "".join([
            "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        ] * 20)),
        ("long_no_repeat_1000", "".join([
            "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        ] * 40)),
    ]

    print(f"{'Name':<25} {'Length':>8} {'Time (ms)':>12} {'Status':<15}")
    print("-" * 70)

    for name, text in test_cases:
        start = time.perf_counter()
        try:
            result, mods = cleaner.clean_repetitions(text)
            elapsed = (time.perf_counter() - start) * 1000

            if elapsed > 1000:
                status = "VERY SLOW"
            elif elapsed > 100:
                status = "SLOW"
            else:
                status = "ok"

            print(f"{name:<25} {len(text):>8} {elapsed:>12.2f} {status}")
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"{name:<25} {len(text):>8} {elapsed:>12.2f} ERROR: {e}")


if __name__ == "__main__":
    print("Starting post-processing performance test...")
    print()

    try:
        test_long_subtitle_edge_case()
        print()
        test_full_postprocessing()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
