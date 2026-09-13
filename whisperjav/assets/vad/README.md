# Bundled Silero VAD models

These ONNX models are shipped inside the WhisperJAV wheel so that the balanced
pipeline's `--vad-version` flag works with **no download at install or run time**
(the owner's requirement for users behind the Great Firewall).

| File | Silero release | Size | md5 |
|---|---|---|---|
| `silero_vad_v3.1.onnx` | snakers4/silero-vad **v3.1** | 0.76 MB | `b41552701dfe751c3dd729b03c0f2ac5` |
| `silero_vad_v4.0.onnx` | snakers4/silero-vad **v4.0** | 1.72 MB | `03da8de2fec4108a089b39f1b4abefef` |
| `silero_vad_v6.2.onnx` | `silero-vad` **6.2.0** PyPI (`silero_vad/data/silero_vad.onnx`) | 2.22 MB | `302cb198a7bb0400c62b73db2942737f` |

All three are MIT licensed — see `LICENSE-silero-vad.txt`.

They are loaded by `whisperjav/modules/silero_vad_adapter.py`, which rebinds
`faster_whisper.vad.get_vad_model` so faster-whisper's own `vad_filter=True` path
runs the selected build. Each generation has a different input signature and a
different output convention; the adapter holds the details.
