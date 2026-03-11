"""
Isolated test: LLM Context Overflow Diagnosis (#196, #212, #132)

This script reproduces the EXACT prompt construction that PySubtrans sends
to the local LLM server, measures token counts, and identifies whether
batches of N subtitle lines fit within an 8K context window.

Usage:
    # Dry run (no server needed) — just measure token counts
    python tests/research/test_llm_context_overflow.py

    # With a real server (Ollama, llama-cpp, etc.)
    python tests/research/test_llm_context_overflow.py --server http://localhost:11434/v1

    # Test specific batch sizes
    python tests/research/test_llm_context_overflow.py --batch-sizes 5,8,10,11,15

Findings will be printed to stdout. No WhisperJAV imports required.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Realistic Japanese subtitle samples (from actual JAV transcription)
# ---------------------------------------------------------------------------

# Mix of short dialogue and long narration lines typical of JAV subtitles
SAMPLE_LINES_JA = [
    "あの、すみません、ちょっとお話があるんですけど",
    "え？何？どうしたの？",
    "実は、昨日の夜のことなんですけど、ちょっと気になることがあって",
    "うん、いいよ。何でも言って",
    "あの、もしかして私のこと好きなんですか？",
    "えっと、その、まあ、そうかもしれないけど",
    "本当に？嬉しい！私もずっとあなたのことが気になってたんです",
    "じゃあ、今度一緒にどこか行かない？",
    "はい、ぜひ！楽しみにしてます",
    "よかった。じゃあ、また連絡するね",
    "ねえ、ちょっと待って。もう少しここにいてもいい？",
    "もちろん。ここにいるよ",
    "ありがとう。なんか安心する",
    "そんなこと言われると照れるな",
    "もう、かわいいんだから",
    "あっ、ダメ、そこは...",
    "大丈夫？痛くない？",
    "ううん、大丈夫。もっとして",
    "こんなに感じてくれるなんて嬉しいよ",
    "もっと強くして。お願い",
    "すごい、こんなに濡れてる",
    "恥ずかしいこと言わないで",
    "でも本当のことだよ",
    "あっ、イキそう... もうダメ",
    "一緒にイこう",
    "あぁっ、イクイクイク！",
    "気持ちよかった？",
    "うん、すごく気持ちよかった。ありがとう",
    "こちらこそ。また会えるかな？",
    "もちろん。いつでも連絡してね",
    # Longer narration lines (worst case for token count)
    "彼女は最初恥ずかしそうにしていたが、次第に大胆になっていき、自分から積極的に求めるようになった",
    "二人は激しく絡み合い、汗だくになりながらも止められない情熱に身を委ねていた",
    "彼の手が彼女の体を優しく撫でると、彼女は甘い声を漏らしながら体をくねらせた",
    "最後の瞬間、二人は同時に絶頂を迎え、しばらくの間、荒い息遣いだけが部屋に響いていた",
    "事が済んだ後、彼女は満足そうな表情で彼の胸に顔を埋め、しばらくそのままの姿勢で横たわっていた",
]

# ---------------------------------------------------------------------------
# 2. Instruction file content (matches standard.txt shipped with WhisperJAV)
# ---------------------------------------------------------------------------

INSTRUCTIONS_STANDARD = Path(__file__).resolve().parents[2] / "whisperjav" / "translate" / "defaults" / "standard.txt"

INSTRUCTIONS_FALLBACK = textwrap.dedent("""\
    ### prompt
    Please translate the following subtitles[ for movie][ to language].

    ### instructions
    You are a professional translator specializing in Japanese translations. Your task is to translate the Japanese movie subtitles into the target language, ensuring they reflect the original meaning as accurately as possible. The goal is to preserve the adult context, lewdness, intimacy, teasing, and pornographic intent of the original dialogue.

    You will receive a batch of lines for translation. Carefully read through the lines, along with any additional context provided.
    Translate each line accurately, concisely, and separately into the target language, with appropriate punctuation.

    The subtitles were AI-generated with Whisper so they are likely to contain transcription errors. If the input seems wrong, try to determine what it should read from the context.

    The translation must have the same number of lines as the original, but you can adapt the content to fit the grammar of the target language.

    Make sure to translate all provided lines and do not ask whether to continue.

    Use any provided context to enhance your translations. If a name list is provided, ensure names are spelled according to the user's preference.

    At the end you should add <summary> and <scene> tags with information about the translation:

    <summary>A one or two line synopsis of the current batch.</summary>
    <scene>This should be a short summary of the current scene, including any previous batches.</scene>

    Your response will be processed by an automated system, so you MUST respond using the required format:

    Example:

    #200
    Original>
    変わりゆく時代において、
    Translation>
    In an ever-changing era,

    #501
    Original>
    進化し続けることが生き残る秘訣です。
    Translation>
    continuing to evolve is the key to survival.

    ### retry_instructions
    There was an issue with the previous translation.

    Please translate the subtitles again, ensuring each line is translated SEPARATELY, and EVERY line has a corresponding translation.

    Do NOT merge lines together in the translation, as this leads to incorrect timings and confusion for the reader.
""")


def load_instructions() -> str:
    """Load the actual instruction file if available, otherwise use fallback."""
    if INSTRUCTIONS_STANDARD.exists():
        return INSTRUCTIONS_STANDARD.read_text(encoding="utf-8")
    return INSTRUCTIONS_FALLBACK


# ---------------------------------------------------------------------------
# 3. Token counting (tiktoken for LLaMA-like BPE, fallback to heuristic)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, otherwise use heuristic.

    For LLaMA/Gemma models, Japanese text is tokenized at ~2-3 BPE tokens
    per character. English is ~1.3 tokens per word. We use cl100k_base
    (GPT-4 tokenizer) as a reasonable proxy.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Heuristic: Japanese chars ~2.5 tokens each, English ~1.3 tokens/word
        ja_chars = sum(1 for c in text if '\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef')
        en_words = len(text.split()) - ja_chars  # rough
        return int(ja_chars * 2.5 + max(0, en_words) * 1.3 + len(text) * 0.05)


def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens across all messages (including role overhead)."""
    total = 0
    for msg in messages:
        total += 4  # role + content framing overhead per message
        total += count_tokens(msg.get("content", ""))
    total += 2  # assistant priming
    return total


# ---------------------------------------------------------------------------
# 4. PySubtrans prompt reconstruction
#    (Reproduces exactly what TranslationPrompt.GenerateMessages() does)
# ---------------------------------------------------------------------------

def build_pysubtrans_messages(
    instructions: str,
    subtitle_lines: list[str],
    context: Optional[dict] = None,
    user_prompt: str = "Translate these subtitles from japanese into english.",
    supports_system_messages: bool = True,
) -> list[dict]:
    """Reconstruct the exact messages PySubtrans sends to the LLM.

    This mirrors TranslationPrompt.GenerateMessages() from PySubtrans 1.5.7.
    """
    messages = []

    # Format subtitle lines as PySubtrans does
    line_prompts = []
    for i, text in enumerate(subtitle_lines, start=1):
        line_prompts.append(f"#{i}\nOriginal>\n{text}\nTranslation>")

    batch_content = "\n\n".join(line_prompts)

    # Add user prompt
    if user_prompt:
        batch_content = f"{user_prompt}\n\n{batch_content}\n"

    # Add context tags if present
    if context:
        context_parts = []
        for tag in ['description', 'names', 'history', 'scene', 'summary', 'batch']:
            if tag in context and context[tag]:
                val = context[tag]
                if isinstance(val, list):
                    val = ", ".join(val)
                context_parts.append(f"<{tag}>{val}</{tag}>")
        if context_parts:
            context_str = "\n".join(context_parts)
            batch_content = f"<context>\n{context_str}\n</context>\n\n{batch_content}\n\n<summary>Summary of the batch</summary>\n<scene>Summary of the scene</scene>\n"

    # Build messages (matching PySubtrans logic)
    if supports_system_messages:
        messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": batch_content})
    else:
        separator = "--------"
        wrapped = f"{separator}\nSYSTEM\n{separator}\n{instructions.strip()}\n{separator}"
        messages.append({"role": "user", "content": f"{wrapped}\n{batch_content}"})

    return messages


def build_expected_response(subtitle_lines: list[str]) -> str:
    """Build the expected LLM response format (for output token estimation)."""
    parts = []
    for i, text in enumerate(subtitle_lines, start=1):
        # Simulate a translation (roughly same length as original in English)
        parts.append(f"#{i}\nTranslation>\nTranslated text for line {i} goes here with some content.")
    parts.append("\n<summary>Summary of the translated batch.</summary>")
    parts.append("<scene>Summary of the current scene including previous context.</scene>")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 5. Context simulation (what PySubtrans sends after batch 1)
# ---------------------------------------------------------------------------

def build_typical_context(scene_num: int = 1, batch_num: int = 2) -> dict:
    """Simulate the context PySubtrans provides for non-first batches.

    After batch 1, PySubtrans sends:
    - scene: summary of current scene so far
    - summary: summary of previous batch
    - batch: "Batch N: summary"
    - history: translated lines from previous batches
    """
    if batch_num <= 1:
        return {}

    return {
        "scene": "The scene takes place in a hotel room. A man and woman are meeting for the first time after connecting online. They start with awkward conversation before becoming intimate.",
        "summary": "The couple had their first intimate encounter. The woman was initially shy but became more bold. They expressed their feelings for each other.",
        "batch": f"Batch {batch_num}: Continuing the intimate scene",
        "history": "1: Excuse me, I have something to tell you\n2: What? What is it?\n3: Actually, about last night, there's something that's been on my mind\n4: Sure, go ahead. Tell me anything\n5: Um, do you perhaps like me?",
    }


# ---------------------------------------------------------------------------
# 6. Main analysis
# ---------------------------------------------------------------------------

def analyze_batch_size(
    batch_size: int,
    instructions: str,
    lines: list[str],
    context: Optional[dict] = None,
    n_ctx: int = 8192,
    max_tokens: Optional[int] = None,
) -> dict:
    """Analyze whether a batch of N lines fits within the context window."""
    actual_lines = lines[:batch_size]

    messages = build_pysubtrans_messages(
        instructions=instructions,
        subtitle_lines=actual_lines,
        context=context,
        supports_system_messages=True,
    )

    # Count input tokens
    input_tokens = count_messages_tokens(messages)

    # Count expected output tokens
    expected_response = build_expected_response(actual_lines)
    output_tokens = count_tokens(expected_response)

    # Total
    total = input_tokens + output_tokens

    # WhisperJAV's cap_batch_size_for_context calculation
    overhead = 2500
    tokens_per_line = 500
    whisperjav_safe_max = max(5, (n_ctx - overhead) // tokens_per_line)

    # Actual available for output
    available_for_output = n_ctx - input_tokens

    result = {
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens_expected": output_tokens,
        "total_tokens": total,
        "n_ctx": n_ctx,
        "fits_in_context": total <= n_ctx,
        "available_for_output": available_for_output,
        "headroom": n_ctx - total,
        "whisperjav_safe_max": whisperjav_safe_max,
        "has_context": context is not None and len(context) > 0,
    }

    if max_tokens:
        result["max_tokens_setting"] = max_tokens
        result["output_would_be_truncated"] = output_tokens > max_tokens

    return result


def print_analysis(result: dict) -> None:
    """Pretty-print analysis result."""
    status = "OK" if result["fits_in_context"] else "OVERFLOW"
    ctx_label = "with context" if result["has_context"] else "no context"

    print(f"  Batch {result['batch_size']:2d} lines ({ctx_label}): "
          f"input={result['input_tokens']:5d} + output={result['output_tokens_expected']:5d} "
          f"= {result['total_tokens']:5d} / {result['n_ctx']} "
          f"[headroom: {result['headroom']:+5d}] "
          f"{'<<< ' + status if not result['fits_in_context'] else status}")


def run_analysis(batch_sizes: list[int], n_ctx: int = 8192) -> None:
    """Run the full analysis."""
    instructions = load_instructions()

    print("=" * 80)
    print("LLM CONTEXT OVERFLOW DIAGNOSIS")
    print("=" * 80)
    print()

    # Token count for instructions alone
    instr_tokens = count_tokens(instructions)
    print(f"Instruction file tokens: {instr_tokens}")
    print(f"Context window: {n_ctx}")
    print()

    # WhisperJAV's calculation
    overhead = 2500
    tokens_per_line = 500
    safe_max = max(5, (n_ctx - overhead) // tokens_per_line)
    print(f"WhisperJAV cap_batch_size_for_context({30}, {n_ctx}) = {safe_max}")
    print(f"  overhead estimate: {overhead}")
    print(f"  per_line estimate: {tokens_per_line}")
    print()

    # --- Test 1: First batch (no context) ---
    print("-" * 80)
    print("TEST 1: First batch (no conversation context)")
    print("-" * 80)
    for bs in batch_sizes:
        if bs > len(SAMPLE_LINES_JA):
            continue
        result = analyze_batch_size(bs, instructions, SAMPLE_LINES_JA, context=None, n_ctx=n_ctx)
        print_analysis(result)

    print()

    # --- Test 2: Subsequent batch (with context from previous batches) ---
    print("-" * 80)
    print("TEST 2: Subsequent batch (WITH conversation context from prior batches)")
    print("-" * 80)
    context = build_typical_context(scene_num=1, batch_num=3)
    for bs in batch_sizes:
        if bs > len(SAMPLE_LINES_JA):
            continue
        result = analyze_batch_size(bs, instructions, SAMPLE_LINES_JA, context=context, n_ctx=n_ctx)
        print_analysis(result)

    print()

    # --- Test 3: Worst case — long narration lines ---
    print("-" * 80)
    print("TEST 3: Worst case — long narration lines + context")
    print("-" * 80)
    # Use the longer lines (indices 30-35)
    long_lines = SAMPLE_LINES_JA[30:]  # The 5 long narration lines
    # Pad with more long lines to test larger batches
    while len(long_lines) < max(batch_sizes):
        long_lines.extend(long_lines[:5])
    for bs in batch_sizes:
        if bs > len(long_lines):
            continue
        result = analyze_batch_size(bs, instructions, long_lines, context=context, n_ctx=n_ctx)
        print_analysis(result)

    print()

    # --- Test 4: Break down where the tokens go ---
    print("-" * 80)
    print("TOKEN BREAKDOWN (batch_size=10, with context)")
    print("-" * 80)
    lines_10 = SAMPLE_LINES_JA[:10]
    context_10 = build_typical_context(scene_num=1, batch_num=3)

    instr_tok = count_tokens(instructions)
    print(f"  Instructions (system message): {instr_tok} tokens")

    # Build user message with context
    line_prompts = [f"#{i}\nOriginal>\n{t}\nTranslation>" for i, t in enumerate(lines_10, 1)]
    batch_text = "\n\n".join(line_prompts)
    batch_tok = count_tokens(batch_text)
    print(f"  Subtitle lines (10 lines, formatted): {batch_tok} tokens")

    user_prompt_tok = count_tokens("Translate these subtitles from japanese into english.")
    print(f"  User prompt: {user_prompt_tok} tokens")

    context_parts = []
    for tag in ['description', 'names', 'history', 'scene', 'summary', 'batch']:
        if tag in context_10 and context_10[tag]:
            val = context_10[tag]
            if isinstance(val, list):
                val = ", ".join(val)
            context_parts.append(f"<{tag}>{val}</{tag}>")
    context_text = "\n".join(context_parts)
    context_tok = count_tokens(context_text)
    print(f"  Context tags (history, scene, summary): {context_tok} tokens")

    template_tok = count_tokens("<context>\n\n</context>\n\n\n\n<summary>Summary of the batch</summary>\n<scene>Summary of the scene</scene>\n")
    print(f"  Template/framing overhead: {template_tok} tokens")

    msg_overhead = 4 * 2 + 2  # 2 messages × 4 + 2 priming
    print(f"  Message framing (role tags etc.): {msg_overhead} tokens")

    total_input = instr_tok + batch_tok + user_prompt_tok + context_tok + template_tok + msg_overhead
    print(f"  --- Total input: ~{total_input} tokens")

    expected_output = build_expected_response(lines_10)
    output_tok = count_tokens(expected_output)
    print(f"  Expected output: {output_tok} tokens")
    print(f"  --- Grand total: ~{total_input + output_tok} tokens")
    print(f"  --- Context window: {n_ctx} tokens")
    print(f"  --- Headroom: {n_ctx - total_input - output_tok:+d} tokens")

    print()

    # --- Recommendations ---
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find the maximum safe batch size
    for bs in range(1, 36):
        if bs > len(SAMPLE_LINES_JA):
            break
        r = analyze_batch_size(bs, instructions, SAMPLE_LINES_JA, context=context, n_ctx=n_ctx)
        if not r["fits_in_context"]:
            print(f"  Maximum safe batch size (with context): {bs - 1}")
            break
    else:
        print(f"  All tested batch sizes fit within {n_ctx} context")

    for bs in range(1, 36):
        if bs > len(SAMPLE_LINES_JA):
            break
        r = analyze_batch_size(bs, instructions, SAMPLE_LINES_JA, context=None, n_ctx=n_ctx)
        if not r["fits_in_context"]:
            print(f"  Maximum safe batch size (no context): {bs - 1}")
            break
    else:
        print(f"  All tested batch sizes fit within {n_ctx} context (no context)")

    print(f"  WhisperJAV's current cap: {safe_max}")
    print()
    print("  NOTE: Token counts use cl100k_base tokenizer (GPT-4). LLaMA/Gemma")
    print("  tokenizers produce ~20-40% MORE tokens for Japanese text due to")
    print("  byte-level BPE. Multiply Japanese token counts by 1.3x for safety.")
    print()


# ---------------------------------------------------------------------------
# 7. Optional: Test against a real server
# ---------------------------------------------------------------------------

def test_with_real_server(server_url: str, batch_size: int = 10, n_ctx: int = 8192) -> None:
    """Send an actual translation request to a running LLM server and observe the result."""
    import requests

    instructions = load_instructions()
    lines = SAMPLE_LINES_JA[:batch_size]
    context = build_typical_context(scene_num=1, batch_num=3)

    messages = build_pysubtrans_messages(
        instructions=instructions,
        subtitle_lines=lines,
        context=context,
        supports_system_messages=True,
    )

    # Compute max_tokens the way WhisperJAV does
    overhead = 2500
    input_per_line_cjk = 300
    output_per_line_en = 120
    output_fixed_tags = 500
    available = n_ctx - overhead - (batch_size * input_per_line_cjk)
    expected = (batch_size * output_per_line_en) + output_fixed_tags
    max_tokens = max(512, min(available, expected * 2))

    print()
    print("=" * 80)
    print(f"LIVE TEST: Sending {batch_size}-line batch to {server_url}")
    print("=" * 80)

    # Check server health
    try:
        health = requests.get(f"{server_url}/models", timeout=5)
        models = health.json().get("data", [])
        print(f"  Server online. Models: {[m['id'] for m in models]}")
    except Exception as e:
        print(f"  ERROR: Cannot reach server at {server_url}: {e}")
        return

    # Send the request
    body = {
        "model": models[0]["id"] if models else "default",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": max_tokens,
        "stream": False,
    }

    print(f"  max_tokens: {max_tokens}")
    print(f"  Sending request...")

    try:
        resp = requests.post(f"{server_url}/chat/completions", json=body, timeout=300)
        data = resp.json()

        if resp.status_code != 200:
            print(f"  ERROR: {resp.status_code} — {data}")
            return

        usage = data.get("usage", {})
        choice = data.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "unknown")
        text = choice.get("message", {}).get("content", "")

        print(f"  Status: {resp.status_code}")
        print(f"  Prompt tokens: {usage.get('prompt_tokens', '?')}")
        print(f"  Completion tokens: {usage.get('completion_tokens', '?')}")
        print(f"  Total tokens: {usage.get('total_tokens', '?')}")
        print(f"  Finish reason: {finish_reason}")
        print(f"  Response length: {len(text)} chars")
        print()

        if finish_reason == "length":
            print("  >>> CONFIRMED: finish_reason='length' — response was TRUNCATED")
            print("  >>> This is the root cause of 'No matches found'")
        elif finish_reason == "stop":
            print("  >>> Response completed normally (finish_reason='stop')")

        # Check if response is parseable by PySubtrans pattern
        import re
        pattern = r'#(\d+)\s+Translation>\s+(.+?)(?=\n#\d|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        print(f"  Parsed translations: {len(matches)} / {batch_size} lines")
        if len(matches) < batch_size:
            print("  >>> SOME LINES MISSING from response — PySubtrans would report 'No matches found'")

    except Exception as e:
        print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Context Overflow Diagnosis")
    parser.add_argument("--server", type=str, default=None,
                        help="URL of a running LLM server (e.g., http://localhost:11434/v1)")
    parser.add_argument("--batch-sizes", type=str, default="5,8,10,11,15,20,25,30",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--n-ctx", type=int, default=8192,
                        help="Context window size (default: 8192)")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    run_analysis(batch_sizes, n_ctx=args.n_ctx)

    if args.server:
        test_with_real_server(args.server, batch_size=10, n_ctx=args.n_ctx)
