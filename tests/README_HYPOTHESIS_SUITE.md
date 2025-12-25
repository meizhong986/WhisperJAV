# Hypothesis Testing Suite - Complete Guide

## What is This?

An automated testing suite for investigating why WhisperJAV v1.7.3 produces ~20% fewer subtitles than v1.7.1. It tests specific parameter hypotheses in isolation to identify the root cause.

## Quick Start

### Option 1: Automated (Recommended)

**Windows:**
```bash
cd tests
run_hypothesis_quickstart.bat
```

**Linux/Mac:**
```bash
cd tests
bash run_hypothesis_quickstart.sh
```

The script will:
1. Validate your setup
2. Check for test audio
3. Guide you through test selection
4. Run tests automatically
5. Show results

### Option 2: Manual

```bash
# 1. Validate setup
python validate_hypothesis_suite.py

# 2. Prepare test audio (25-minute subset)
ffmpeg -i video.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le -ar 16000 -ac 1 subset.wav

# 3. Run quick test
python hypothesis_test_suite.py --audio subset.wav --quick

# 4. Review results
cat hypothesis_results.json
```

## File Organization

```
tests/
├── Core Files
│   ├── hypothesis_configs.py          # Test configuration definitions
│   ├── hypothesis_test_suite.py       # Main test orchestrator
│   ├── inspect_hypothesis_configs.py  # Configuration inspection utility
│   └── validate_hypothesis_suite.py   # Setup validation
│
├── Quick Start Scripts
│   ├── run_hypothesis_quickstart.sh   # Interactive script (Linux/Mac)
│   └── run_hypothesis_quickstart.bat  # Interactive script (Windows)
│
└── Documentation
    ├── README_HYPOTHESIS_SUITE.md     # This file (complete guide)
    ├── HYPOTHESIS_TESTING_README.md   # User manual
    ├── HYPOTHESIS_SUITE_SUMMARY.md    # Technical overview
    └── ADDING_NEW_HYPOTHESES.md       # Developer guide
```

## Which Documentation Do I Need?

| If you want to... | Read this |
|-------------------|-----------|
| **Get started quickly** | This file (README_HYPOTHESIS_SUITE.md) |
| **Understand test parameters** | HYPOTHESIS_TESTING_README.md |
| **Add new hypotheses** | ADDING_NEW_HYPOTHESES.md |
| **Understand architecture** | HYPOTHESIS_SUITE_SUMMARY.md |
| **Inspect configurations** | Run `python inspect_hypothesis_configs.py --help` |

## What Gets Tested?

### The Problem
v1.7.3 produces 128 fewer subtitles than v1.7.1 (657 → 529)

### The Hypotheses

1. **VAD Parameter Wiring**: Are `speech_pad_ms` and `min_silence_duration_ms` being applied?
2. **ASR Duration Filtering**: Do `no_speech_threshold` and `logprob_threshold` filter short segments?
3. **Temperature Fallback**: Does temperature configuration affect detection?
4. **Patience/Beam Search**: Do beam search parameters affect quality?

### The Tests

**Quick Mode** (5 tests, ~15 minutes):
- Baseline
- VAD both zeros
- ASR balanced thresholds
- Temperature balanced fallback
- Beam balanced combo

**Full Suite** (18 tests, ~54 minutes):
- All variations of all 4 hypotheses

## Understanding Results

### Summary Table

```
Config                         Hypothesis            Subs   Duration       Avg   <1s   >5s
v1.7.3_baseline                baseline                55      287.3s    5.22s    12     8
vad_both_zeros                 vad_params              64 (+9)  324.5s    5.07s    16     8
```

**What to look for:**
- **Positive delta** (+9): Config recovered 9 missing subtitles ✓
- **Negative delta** (-3): Config made it worse ✗
- **No change** (0): Config had no effect

### JSON Results

Detailed metrics in `hypothesis_results.json`:
```json
{
  "baseline": { "total_subtitles": 55 },
  "results": [
    {
      "config_name": "vad_both_zeros",
      "total_subtitles": 64,
      "comparison": {
        "subtitle_delta_vs_baseline": 9,
        "improvement_vs_baseline_pct": 16.4
      }
    }
  ],
  "recommendations": [
    "Most impactful hypothesis: vad_params (+9 subtitles)"
  ]
}
```

### Next Steps After Testing

If a hypothesis shows improvement:

1. **Validate**: Run full video with winning config
2. **Compare**: Manually review recovered subtitles
3. **Investigate**: Why does this parameter have this effect?
4. **Fix**: Update defaults or fix parameter wiring bug

## Common Use Cases

### I want to quickly test if VAD parameters are the issue
```bash
python hypothesis_test_suite.py --audio subset.wav --hypothesis vad_params
```

### I want to test all hypotheses thoroughly
```bash
python hypothesis_test_suite.py --audio subset.wav --reference v1.7.1.srt
```

### I want to understand what a config does
```bash
python inspect_hypothesis_configs.py --hypothesis vad_params --detailed
```

### I want to compare two configurations
```bash
python inspect_hypothesis_configs.py --compare v1.7.3_baseline vad_both_zeros
```

### I want to add my own hypothesis
See `ADDING_NEW_HYPOTHESES.md`

## Troubleshooting

### "validation failed"
Your environment may be missing dependencies. Check error messages.

### "Audio file not found"
Ensure you've created `subset.wav` in the current directory.

### "Out of memory"
Use shorter audio (10 minutes instead of 25) or CPU mode.

### Tests produce identical results
Either:
- Baseline is already optimal, OR
- Parameters aren't wired correctly (need code investigation)

### Permission denied on scripts
```bash
chmod +x run_hypothesis_quickstart.sh
```

## Performance

| Test Mode | Tests | Time | Best For |
|-----------|-------|------|----------|
| Quick | 5 | ~15 min | Initial investigation |
| Per-hypothesis | 3-5 | ~10 min | Focused testing |
| Full suite | 18 | ~54 min | Comprehensive analysis |

*Timing based on 25-minute audio with GPU acceleration*

## Test Audio Requirements

**Recommended**:
- Duration: 25 minutes (from problematic time range)
- Format: WAV, 16kHz, mono, 16-bit PCM
- Source: Extract from 1h52m to 2h17m (highest missing subtitle density)

**Minimum**:
- Duration: 10 minutes (shorter = faster but less reliable)
- Any audio format supported by ffmpeg

**Command**:
```bash
ffmpeg -i video.mp4 -ss 01:52:00 -t 00:25:00 -vn -acodec pcm_s16le -ar 16000 -ac 1 subset.wav
```

## Advanced Usage

### Skip baseline if already run
```bash
python hypothesis_test_suite.py --audio subset.wav --skip-baseline --hypothesis vad_params
```

### Custom output location
```bash
python hypothesis_test_suite.py --audio subset.wav \
    --output-dir ./my_outputs \
    --temp-dir ./my_temp \
    --results-file ./my_results.json
```

### Export configs to JSON
```bash
python inspect_hypothesis_configs.py --json all_configs.json
```

### Show statistics
```bash
python inspect_hypothesis_configs.py --stats
```

## Design Philosophy

1. **Isolation**: Each test modifies ONLY the parameter under investigation
2. **Reproducibility**: Same config → same results
3. **Traceability**: Every parameter change is logged
4. **Comparison**: All results compared to baseline and reference
5. **Modularity**: Easy to add new hypotheses

## FAQ

**Q: Do I need a reference v1.7.1 SRT?**
A: No, it's optional. But it helps validate that your test audio contains the regression.

**Q: Can I run just one specific test?**
A: Not directly, but you can create a custom hypothesis with just that test.

**Q: Why does the test take so long?**
A: Each test runs full transcription pipeline. Use `--quick` for faster iteration.

**Q: Can I test on the full video?**
A: Yes, but it will take hours. Use a subset for hypothesis testing, then validate winners on full video.

**Q: What if no hypothesis shows improvement?**
A: The issue may be in untested parameters, or in code logic rather than config values.

## Getting Help

1. **Check validation**: `python validate_hypothesis_suite.py`
2. **Check documentation**: Read relevant .md file above
3. **Check configs**: `python inspect_hypothesis_configs.py --detailed`
4. **Check logs**: Review console output for errors

## Contributing

Found the root cause? Please:
1. Document findings in test results
2. Add comments explaining the issue
3. Consider submitting a PR with the fix

## Credits

**Created by**: Claude Code (Senior Test Automation Engineer)
**For**: ASR Regression Investigation v1.7.3
**Date**: December 2024
**License**: Same as WhisperJAV

---

**Ready to start?** Run `python validate_hypothesis_suite.py` to check your setup!
