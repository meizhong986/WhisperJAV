Thanks for the detailed logs @destinyawaits, and @zhstark for confirming the same on Ubuntu.

**Root cause confirmed.** The v1.8.6 auto-cap formula reduced your batch size from 30 to 17, but 17 lines is still too many for an 8K context window with long Japanese subtitle lines. The prompt (instructions + subtitles + formatting) overflows the context, causing the "Hit API token limit" error. The regex failure is a downstream symptom — the model produces garbled output when pushed past its context limit.

**Immediate workaround — set a smaller batch size:**

```bash
whisperjav-translate -i your_file.srt --provider local --max-batch-size 10
```

Or via the interactive setup:
```bash
whisperjav-translate --configure
# When prompted for "Max batch size", enter 10
```

For 8K context models like gemma-9b, a batch size of 10-12 is the safe range. The default 30 is designed for cloud APIs with 128K+ context windows.

**Fix:** We've updated the auto-cap formula to be more conservative — 8K context will now auto-reduce to 11 (was 17). This will ship in the next release so you won't need the manual override.
