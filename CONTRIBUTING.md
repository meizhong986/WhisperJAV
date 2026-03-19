# Contributing to WhisperJAV

Thank you for your interest in contributing to WhisperJAV! This guide covers how to contribute to the project.

---

## Reporting Issues

- Search [existing issues](https://github.com/meizhong986/whisperjav/issues) before creating a new one
- Include your system details: OS, GPU model, Python version, WhisperJAV version
- Attach error logs (`whisperjav.log` or console output) when reporting bugs
- For installation issues, include the install log (`install_log_v*.txt`)

---

## Improving Chinese Translations

The documentation is available in English and Simplified Chinese. Chinese translations are generated with AI and reviewed by maintainers. Community improvements are welcome and encouraged.

### How to Contribute

1. Browse the Chinese docs at [meizhong986.github.io/WhisperJAV/zh/](https://meizhong986.github.io/WhisperJAV/zh/)
2. Find a page that needs improvement
3. Click the edit icon (pencil) on the page, or edit the file directly in `docs/zh/`
4. Submit a Pull Request with the `translation` label

### Translation Guidelines

- **Use the glossary**: See [`docs/translation-glossary.md`](docs/translation-glossary.md) for consistent terminology
- **Keep in English**: Code blocks, CLI commands, file paths, product names, model names, CLI flags, pipeline names
- **Translate**: Prose text, section headings, table headers, descriptive table cells, admonition titles
- **Tone**: Natural, readable Chinese. Avoid overly literal or machine-translation style.
- **Format**: Preserve markdown formatting exactly (headers, tables, code fences, admonitions)

### File Structure

```
docs/
  en/          # English source (authoritative)
  zh/          # Chinese translations
  translation-glossary.md   # Term consistency reference
```

English docs in `docs/en/` are the source of truth. Chinese docs in `docs/zh/` should mirror the same structure and content.

### Automated Translation Sync

When English docs change on `main`, a GitHub Action can automatically translate the changes and create a PR for review. This requires the `ANTHROPIC_API_KEY` secret and the `ENABLE_AUTO_TRANSLATE` repository variable set to `true`.

---

## Code Contributions

### Development Setup

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
python install.py --dev
```

### Code Style

- Run `ruff check whisperjav/` before submitting
- Run `ruff format whisperjav/` for formatting
- Follow existing code patterns and conventions

### Testing

```bash
python -m pytest tests/ -v
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run linting and tests
4. Submit a PR with a clear description of what and why
