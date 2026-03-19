#!/usr/bin/env python3
"""
Translate changed English documentation to Chinese using the Anthropic API.

Usage:
    # Translate specific files
    python scripts/translate_docs.py docs/en/faq.md docs/en/index.md

    # Translate all changed files (auto-detect via git diff)
    python scripts/translate_docs.py --changed

    # Translate ALL English files (full rebuild)
    python scripts/translate_docs.py --all

    # Dry run (show what would be translated, don't call API)
    python scripts/translate_docs.py --changed --dry-run

Requirements:
    pip install anthropic

Environment:
    ANTHROPIC_API_KEY - Required. Your Anthropic API key.
    TRANSLATION_MODEL - Optional. Default: claude-sonnet-4-20250514
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).parent.parent / "docs"
EN_DIR = DOCS_ROOT / "en"
ZH_DIR = DOCS_ROOT / "zh"
GLOSSARY_FILE = DOCS_ROOT / "translation-glossary.md"

DEFAULT_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """\
You are a professional technical translator specializing in software documentation.
Translate the following English documentation to Simplified Chinese (简体中文).

RULES:
1. Keep ALL code blocks, CLI commands, file paths, and environment variables in English.
2. Keep product names in English: WhisperJAV, Whisper, Faster-Whisper, PyTorch, CUDA, \
FFmpeg, WebView2, Ollama, conda, pip, Git, NVIDIA, Ubuntu, Fedora, Arch Linux, macOS, Homebrew.
3. Keep model names in English: large-v3, large-v2, turbo, medium, small, anime-whisper, Kotoba.
4. Keep provider names in English: DeepSeek, Gemini, Claude, GPT, Ollama, OpenRouter.
5. Keep CLI flags in English: --mode, --sensitivity, --translate, --force-cuda, etc.
6. Keep pipeline names as proper nouns: Balanced, Fast, Faster, Fidelity, Transformers, \
ChronosJAV, Qwen3-ASR.
7. Translate all prose text, section headings, table headers, and descriptive table cells.
8. For admonitions (e.g., !!! tip "Title"), keep the type keyword in English, translate the title.
9. Keep link targets as-is, translate link text.
10. Preserve markdown formatting exactly (headers, tables, code fences, admonitions, images).
11. Use natural, readable Chinese — not machine-translation style.
12. Do NOT add or remove content. Translate what is there.
13. Do NOT include manual Table of Contents sections with CJK anchor links \
(mkdocs sidebar handles navigation).

OUTPUT: Return ONLY the translated markdown. No preamble, no explanation.
"""


def load_glossary() -> str:
    """Load the translation glossary for inclusion in the prompt."""
    if GLOSSARY_FILE.exists():
        return GLOSSARY_FILE.read_text(encoding="utf-8")
    return ""


def get_changed_files(base_ref: str = "HEAD~1") -> list[str]:
    """Get list of changed English doc files via git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "--", "docs/en/"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.endswith(".md")]
    except subprocess.CalledProcessError:
        print("Warning: git diff failed. Falling back to empty list.", file=sys.stderr)
        return []


def get_all_english_files() -> list[str]:
    """Get all English markdown files."""
    return [
        str(p.relative_to(Path.cwd()))
        for p in EN_DIR.rglob("*.md")
    ]


def en_to_zh_path(en_path: str) -> Path:
    """Convert an English doc path to the corresponding Chinese path."""
    # docs/en/foo/bar.md -> docs/zh/foo/bar.md
    return Path(en_path.replace("docs/en/", "docs/zh/", 1))


def translate_file(en_path: str, model: str, dry_run: bool = False) -> bool:
    """Translate a single English file to Chinese."""
    en_file = Path(en_path)
    zh_file = en_to_zh_path(en_path)

    if not en_file.exists():
        print(f"  SKIP {en_path} (file not found)")
        return False

    en_content = en_file.read_text(encoding="utf-8")
    if not en_content.strip():
        print(f"  SKIP {en_path} (empty file)")
        return False

    if dry_run:
        print(f"  WOULD translate {en_path} -> {zh_file}")
        return True

    print(f"  Translating {en_path} -> {zh_file} ...", end=" ", flush=True)

    try:
        import anthropic
    except ImportError:
        print("\nError: 'anthropic' package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    glossary = load_glossary()

    user_message = f"""## Translation Glossary

{glossary}

## Source Document

{en_content}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        zh_content = response.content[0].text
    except Exception as e:
        print(f"FAILED ({e})")
        return False

    # Write output
    zh_file.parent.mkdir(parents=True, exist_ok=True)
    zh_file.write_text(zh_content, encoding="utf-8")

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    print(f"OK ({tokens_in}+{tokens_out} tokens)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Translate WhisperJAV English docs to Chinese"
    )
    parser.add_argument("files", nargs="*", help="Specific files to translate")
    parser.add_argument(
        "--changed",
        action="store_true",
        help="Auto-detect changed files via git diff",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Translate ALL English files",
    )
    parser.add_argument(
        "--base-ref",
        default="HEAD~1",
        help="Git ref for diff base (default: HEAD~1)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRANSLATION_MODEL", DEFAULT_MODEL),
        help=f"Anthropic model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be translated without calling API",
    )
    args = parser.parse_args()

    # Determine files to translate
    if args.all:
        files = get_all_english_files()
    elif args.changed:
        files = get_changed_files(args.base_ref)
    elif args.files:
        files = args.files
    else:
        parser.error("Specify files, --changed, or --all")
        return

    if not files:
        print("No files to translate.")
        return

    print(f"Translating {len(files)} file(s) with {args.model}:")

    success = 0
    failed = 0
    for f in sorted(files):
        if translate_file(f, args.model, args.dry_run):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} translated, {failed} skipped/failed.")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
