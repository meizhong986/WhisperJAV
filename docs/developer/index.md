# Developer Documentation

Technical documentation for contributors and developers.

---

## Architecture

- [Config Sources Hierarchy](../architecture/CONFIG_SOURCES_HIERARCHY.md) — understanding the configuration priority system

## Architecture Decision Records

- [ADR-001: YAML Config Architecture](../adr/ADR-001-yaml-config-architecture.md)
- [ADR-002: Batched Transcription Pipeline](../adr/ADR-002-batched-transcription-pipeline.md)
- [ADR-003: Qwen3-ASR Integration](../architecture/ADR-003-qwen3-asr-integration.md)
- [ADR-004: Dedicated Qwen Pipeline](../architecture/ADR-004-dedicated-qwen-pipeline.md)

## Development Setup

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Code Quality

```bash
python -m ruff check whisperjav/
python -m ruff format whisperjav/
```

## Building the Installer

See the [CLAUDE.md](https://github.com/meizhong986/whisperjav/blob/main/CLAUDE.md) file for full build instructions.
