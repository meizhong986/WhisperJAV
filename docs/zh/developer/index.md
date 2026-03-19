# 开发者文档

面向贡献者和开发者的技术文档。

---

## 架构

- [配置来源层级](../architecture/CONFIG_SOURCES_HIERARCHY.md) — 了解配置优先级体系

## 架构决策记录

- [ADR-001：YAML 配置架构](../adr/ADR-001-yaml-config-architecture.md)
- [ADR-002：批量转录处理管线](../adr/ADR-002-batched-transcription-pipeline.md)
- [ADR-003：Qwen3-ASR 集成](../architecture/ADR-003-qwen3-asr-integration.md)
- [ADR-004：专用 Qwen 处理管线](../architecture/ADR-004-dedicated-qwen-pipeline.md)

## 开发环境搭建

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
pip install -e ".[dev]"
```

## 运行测试

```bash
python -m pytest tests/ -v
```

## 代码质量

```bash
python -m ruff check whisperjav/
python -m ruff format whisperjav/
```

## 构建安装程序

完整的构建说明请参阅 [CLAUDE.md](https://github.com/meizhong986/whisperjav/blob/main/CLAUDE.md) 文件。
