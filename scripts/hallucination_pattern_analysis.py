"""
Hallucination Pattern Analysis
Identifies known hallucination patterns in missing vs matched subtitles
"""

import re
import sys
from collections import defaultdict

# Force UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def parse_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds"""
    time_part, ms_part = timestamp.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_srt_file(filepath: str):
    """Parse SRT file into list of (index, start, end, text)"""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
            timestamp_line = lines[1].strip()
            text = '\n'.join(lines[2:]).strip()

            start_str, end_str = timestamp_line.split(' --> ')
            start_time = parse_timestamp(start_str.strip())
            end_time = parse_timestamp(end_str.strip())

            entries.append((index, start_time, end_time, text))
        except:
            continue

    return entries


# Known hallucination patterns from Japanese Whisper models
HALLUCINATION_PATTERNS = [
    # Generic thank you messages (common hallucination)
    r'ご視聴.*ありがとう',
    r'チャンネル登録',
    r'高評価.*お願い',
    r'コメント.*お待ち',
    r'次回.*動画.*お会い',
    r'次回.*ビデオ.*お会い',
    r'最後まで.*ご覧.*ありがとう',
    r'購読.*ボタン',

    # Nonsensical technical phrases (common hallucination)
    r'電子レンジ.*600w',
    r'電子レンジ.*電源',
    r'電子レンジ.*電流',
    r'電子レンジ.*回転',
    r'電気機関車.*電流',
    r'スマートフォン.*電源',
    r'スイッチ.*外す',

    # Generic family/love messages (common hallucination)
    r'私たちの家族',
    r'あなた.*愛して',
    r'私.*あなた.*大好き',
    r'私.*感謝',
    r'幸運.*与えられ',
    r'幸せ.*です',

    # Subtitle branding (likely added by subtitle creator, not from audio)
    r'字幕.*by',
    r'字幕.*制作',
    r'翻訳.*by',
]


def is_hallucination(text: str) -> bool:
    """Check if text matches known hallucination patterns"""
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def analyze_hallucinations():
    ref_file = r"C:\BIN\git\WhisperJav_V1_Minami_Edition\test_results\1_7_1-vs-1_7_3\HODV-22019.1_7_1.srt"
    result_file = r"C:\BIN\git\WhisperJav_V1_Minami_Edition\test_results\1_7_1-vs-1_7_3\1.7.3-f\HODV-22019.ja.pass1.srt"

    print("HALLUCINATION PATTERN ANALYSIS")
    print("="*80)

    ref_entries = parse_srt_file(ref_file)
    result_entries = parse_srt_file(result_file)

    # Analyze hallucinations in each file
    ref_halluc = [(idx, start, end, text) for idx, start, end, text in ref_entries if is_hallucination(text)]
    result_halluc = [(idx, start, end, text) for idx, start, end, text in result_entries if is_hallucination(text)]

    print(f"\nv1.7.1: {len(ref_halluc)} hallucinations out of {len(ref_entries)} total ({len(ref_halluc)/len(ref_entries)*100:.1f}%)")
    print(f"v1.7.3: {len(result_halluc)} hallucinations out of {len(result_entries)} total ({len(result_halluc)/len(result_entries)*100:.1f}%)")

    # Show v1.7.1 hallucinations
    print("\n" + "="*80)
    print("v1.7.1 HALLUCINATIONS (first 30)")
    print("="*80)
    for idx, start, end, text in ref_halluc[:30]:
        print(f"[{start:>8.1f}s] #{idx:<4} {text[:70]}")

    # Show v1.7.3 hallucinations
    print("\n" + "="*80)
    print("v1.7.3 HALLUCINATIONS (first 30)")
    print("="*80)
    for idx, start, end, text in result_halluc[:30]:
        print(f"[{start:>8.1f}s] #{idx:<4} {text[:70]}")

    # Timeline comparison
    print("\n" + "="*80)
    print("TIMELINE COMPARISON (first 20 minutes)")
    print("="*80)

    print("\nv1.7.1 timeline:")
    for idx, start, end, text in ref_entries[:30]:
        halluc_marker = " [HALLUC]" if is_hallucination(text) else ""
        print(f"  {start:>6.0f}s: {text[:50]}{halluc_marker}")

    print("\nv1.7.3 timeline:")
    for idx, start, end, text in result_entries[:30]:
        halluc_marker = " [HALLUC]" if is_hallucination(text) else ""
        print(f"  {start:>6.0f}s: {text[:50]}{halluc_marker}")

    # Pattern frequency
    print("\n" + "="*80)
    print("HALLUCINATION PATTERN FREQUENCY")
    print("="*80)

    ref_pattern_counts = defaultdict(int)
    result_pattern_counts = defaultdict(int)

    for _, _, _, text in ref_entries:
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, text):
                ref_pattern_counts[pattern] += 1

    for _, _, _, text in result_entries:
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, text):
                result_pattern_counts[pattern] += 1

    all_patterns = set(ref_pattern_counts.keys()) | set(result_pattern_counts.keys())

    print(f"\n{'Pattern':<40} {'v1.7.1':<10} {'v1.7.3':<10} {'Reduction'}")
    print("-"*80)

    for pattern in sorted(all_patterns, key=lambda p: ref_pattern_counts[p], reverse=True):
        ref_count = ref_pattern_counts[pattern]
        result_count = result_pattern_counts[pattern]
        reduction = ref_count - result_count
        reduction_pct = (reduction / ref_count * 100) if ref_count > 0 else 0

        print(f"{pattern:<40} {ref_count:<10} {result_count:<10} -{reduction} ({reduction_pct:.0f}%)")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    halluc_reduction = len(ref_halluc) - len(result_halluc)
    total_reduction = len(ref_entries) - len(result_entries)

    print(f"\n1. Total subtitle reduction: {total_reduction} ({len(ref_entries)} → {len(result_entries)})")
    print(f"2. Hallucination reduction: {halluc_reduction} ({len(ref_halluc)} → {len(result_halluc)})")
    print(f"3. Percentage of reduction explained by hallucination removal: {halluc_reduction/total_reduction*100:.1f}%")

    # Time coverage
    ref_first_real = None
    for idx, start, end, text in ref_entries:
        if not is_hallucination(text):
            ref_first_real = (idx, start, text)
            break

    result_first_real = None
    for idx, start, end, text in result_entries:
        if not is_hallucination(text):
            result_first_real = (idx, start, text)
            break

    print(f"\n4. First non-hallucination subtitle:")
    if ref_first_real:
        print(f"   v1.7.1: #{ref_first_real[0]} at {ref_first_real[1]:.1f}s: {ref_first_real[2][:40]}")
    if result_first_real:
        print(f"   v1.7.3: #{result_first_real[0]} at {result_first_real[1]:.1f}s: {result_first_real[2][:40]}")

    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_hallucinations()
