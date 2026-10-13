"""Loads both measuring instruments in one process: the speech segmenter recorder (segmenter_recorder, WJ_RECORD_OUT)
and the call tracer (call_tracer, WJ_TRACE_OUT). Python imports only one module named sitecustomize, so this one runs
the other two from their own files. Neither changes any value; see their docstrings."""
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for _name, _folder in (("wj_segmenter_recorder", "segmenter_recorder"), ("wj_call_tracer", "call_tracer")):
    _spec = importlib.util.spec_from_file_location(_name, _HERE / _folder / "sitecustomize.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
