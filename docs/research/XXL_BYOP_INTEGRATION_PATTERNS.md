# XXL BYOP Integration Patterns — Subtitle Edit & pyvideotrans

> Created: 2026-03-15 | Source: R2 research (SubtitleEdit C#, pyvideotrans Python)

## Goal

Understand how existing applications integrate Faster Whisper XXL as an external executable, to inform WhisperJAV's BYOP (Bring Your Own Provider) MVP implementation.

---

## 1. pyvideotrans Integration (Python — Most Relevant)

### Executable Discovery
- **Pure BYOP**: User selects exe path via file dialog
- Stored in `config.settings['Faster_Whisper_XXL']`
- No auto-download, no validation beyond file existence

### Command Construction

```python
cmd = [
    config.settings.get('Faster_Whisper_XXL', ''),  # exe path
    audio_file,                                       # input file
    "-pp",                                            # print progress
    "-f", "srt"                                       # output format
]
if language != 'auto':
    cmd.extend(['-l', language.split('-')[0]])
if initial_prompt:
    cmd.extend(['--initial_prompt', prompt])
cmd.extend(['--model', model_name, '--output_dir', target_dir])

# Extra user args from pyvideotrans.txt file next to exe
txt_file = Path(exe_path).parent / 'pyvideotrans.txt'
if txt_file.exists():
    cmd.extend(txt_file.read_text(encoding='utf-8').strip().split(' '))
```

### Subprocess Execution

```python
subprocess.run(
    cmd_list,
    capture_output=True,
    text=True,
    check=True,
    encoding='utf-8',
    creationflags=0,
    cwd=os.path.dirname(cmd_list[0])  # cwd = exe's directory
)
```

- **Blocking** `subprocess.run` — waits for completion
- No real-time progress streaming despite passing `-pp`
- `check=True` raises `CalledProcessError` on non-zero exit

### Output Collection
- Does NOT parse stdout
- Relies entirely on the SRT file written by XXL to `--output_dir`
- Output filename: `{input_stem}.srt`
- After completion: `shutil.copy2(output_srt, target_location)`

### Error Handling
- `CalledProcessError` → `RuntimeError(e.stderr)`
- No retry logic, no special error detection

### Key Observations
- ~20 lines of actual integration code
- Pre-converts audio to 16kHz WAV (unnecessary — XXL handles extraction)
- Extra args via text file next to exe — clever BYOP pattern
- No progress reporting to user during transcription

---

## 2. Subtitle Edit Integration (C# — More Sophisticated)

### Executable Discovery
- Exe name: `faster-whisper-xxl.exe` (Windows) / `faster-whisper-xxl` (Linux)
- Located in: `{WhisperFolder}/Purfview-Faster-Whisper-XXL/`
- SE manages downloading and extracting from GitHub releases
- Models stored in: `{WhisperFolder}/Purfview-Faster-Whisper-XXL/_models/faster-whisper-{modelName}/`

### Command Construction

```
faster-whisper-xxl.exe --language {lang} --model "{model}" {customArgs} "{inputFile}"
```

- Default custom args: `"--standard --beep_off"` (from settings)
- `--standard_asia` available as alternative for Asian languages
- Can pass video files directly (mkv/mp4) — no pre-extraction needed
- No `--output_dir` or `-f` used by SE (relies on stdout parsing)

### Output Capture — Two Paths

**1. Stdout (real-time):** Parses timestamped lines as they arrive:
```
[MM:SS.mmm --> MM:SS.mmm] text
[HH:MM:SS.mmm --> HH:MM:SS.mmm] text
```

**2. File-based (fallback):** After process exits, looks for `{inputFile}.srt`, `.vtt`, `.json` in whisper folder.

### Progress Capture
- `-pp` flag enables tqdm-style progress in stderr
- Parses: `whisper_full: progress = XX%` or `XX%|...` patterns

### Error Detection
String matching on stderr/stdout:
- `"error: unknown argument"`
- `"CUDA failed with error out of memory"`
- `"not all tensors loaded from model file"`

### Environment Variables
```
PYTHONIOENCODING=utf-8
PYTHONUTF8=1
PATH extended with FFmpeg + whisper folder
Working directory = whisper folder
```

---

## 3. Common Minimal Pattern

Both applications treat XXL as a black box:

```
Call exe → Wait → Read SRT file
```

### Minimum Required Arguments

| Arg | Purpose | Notes |
|-----|---------|-------|
| positional `input` | Input file | Video or audio — XXL handles extraction |
| `--model` / `-m` | Model name | e.g. `large-v3` |
| `--language` / `-l` | Language code | e.g. `ja`, or omit for auto |
| `--output_dir` / `-o` | Where to write SRT | If omitted, writes next to input |

### Recommended Additional Arguments

| Arg | Purpose |
|-----|---------|
| `--beep_off` | Suppress completion beep |
| `-pp` | Progress output (parseable from stderr) |
| `--standard_asia` | Good default for Japanese content |

---

## 4. Proposed MVP for WhisperJAV

### Pseudocode

```python
import subprocess
import os
from pathlib import Path

def run_xxl(
    input_file: str,
    exe_path: str,
    model: str = "large-v3",
    language: str = "ja",
    output_dir: str = None,
    extra_args: str = "",      # User passthrough (e.g. "--standard_asia --ff_vocal_extract mdx_kim2")
) -> Path:
    """
    Call faster-whisper-xxl as subprocess, return path to generated SRT.
    BYOP: user provides exe, configures XXL flags themselves.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="whisperjav_xxl_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        exe_path,
        input_file,
        "--model", model,
        "--language", language,
        "--output_dir", output_dir,
        "--beep_off",
    ]

    # Passthrough user args
    if extra_args:
        cmd.extend(extra_args.split())

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(Path(exe_path).parent),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )

    if result.returncode != 0:
        raise RuntimeError(f"XXL failed (exit {result.returncode}):\n{result.stderr}")

    srt_path = Path(output_dir) / f"{Path(input_file).stem}.srt"
    if not srt_path.is_file():
        raise RuntimeError(f"XXL completed but SRT not found at {srt_path}")

    return srt_path
```

### GUI Integration (Ensemble Mode)

In ensemble mode, XXL would be one of the "passes":

1. **Settings panel**: File picker for exe path + text field for extra args
2. **Ensemble pass config**: User selects "XXL Faster-Whisper" as one pass
3. **Execution**: WhisperJAV calls `run_xxl()` → gets SRT → feeds into ensemble merge
4. **Merge**: Standard ensemble merge logic combines XXL output with other passes

### What WhisperJAV Does NOT Do (MVP)

- No model management (user handles model downloads via XXL itself)
- No VAD, scene detection, or speech enhancement for XXL pass (XXL does its own)
- No stdout parsing for progress (wait for completion)
- No validation of extra_args (pure passthrough)
- No integration with sensitivity profiles or transcription tuner
- No special error recovery

### What the User Configures

| Setting | Where | Example |
|---------|-------|---------|
| XXL exe path | GUI file picker / `--xxl-exe` CLI | `C:\Tools\faster-whisper-xxl.exe` |
| Model | GUI dropdown / `--xxl-model` CLI | `large-v3` |
| Extra args | GUI text field (persisted to asr_config.json) | `--standard_asia --ff_vocal_extract mdx_kim2 --compute_type float16` |

### Estimated Effort

- XXL runner function: ~50 lines
- GUI integration (file picker + args field): ~30 lines
- Ensemble pass registration: ~20 lines
- **Total**: ~100 lines, no architectural changes

---

## 5. pyvideotrans Extra Args Pattern (Worth Adopting)

pyvideotrans loads extra args from a `pyvideotrans.txt` file next to the exe. This is a nice BYOP pattern — users can configure XXL without touching WhisperJAV's config:

```python
txt_file = Path(exe_path).parent / 'pyvideotrans.txt'
if txt_file.exists():
    cmd.extend(txt_file.read_text(encoding='utf-8').strip().split())
```

WhisperJAV could adopt a similar pattern (e.g., `whisperjav_xxl.txt` next to the exe) in addition to the GUI text field.
